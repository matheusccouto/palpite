""" Data requesting and wrangling. """

import json
import os
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd
import requests

THIS_FOLDER = os.path.dirname(__file__)


def request_to_df(url: str, key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """ Create dataframe from request data. """
    data = requests.get(url, verify=False, **kwargs).json()

    # If specified some inner key.
    if key is not None:
        data = data[key]

    # If it is a dictionary, use the values method.
    if isinstance(data, dict):
        data = data.values()

    return pd.DataFrame(data)


class CartolaFCAPI:
    """ A high level wrapper for the Cartola FC API. """

    host = r"https://api.cartolafc.globo.com/"

    def clubs(self) -> pd.DataFrame:
        """ Get clubs data frame. """
        return request_to_df(self.host + r"clubes").set_index("id")

    def matches(self) -> pd.DataFrame:
        """ Get next matches data frame. """
        return request_to_df(self.host + r"partidas", "partidas")

    def players(self) -> pd.DataFrame:
        """ Get players data frame. """
        return request_to_df(self.host + r"atletas/mercado", "atletas").set_index(
            "atleta_id"
        )

    def schemes(self):
        """ Get schemes data frame. """
        return request_to_df(self.host + r"esquemas")

    def positions(self) -> pd.DataFrame:
        """ Get positions data frame. """
        return request_to_df(self.host + r"atletas/mercado", "posicoes")

    def status(self) -> pd.DataFrame:
        """ Get status data frame. """
        return request_to_df(self.host + r"atletas/mercado", "status")


class TheOddsAPI:
    """ A high level wrapper for the-odds-api.com. User must provide a private key. """

    host = r"https://api.the-odds-api.com/v3/"

    def __init__(self, key: str, cache_folder: Optional[str] = None):
        self.key = key
        self.cache_folder = "cache" if cache_folder is None else cache_folder

    @staticmethod
    def clean_betting_lines(data: pd.DataFrame) -> pd.DataFrame:
        """ Clean betting lines dataframe. """
        # Remove entries with no odds.
        data = data[data["sites_count"] > 0]

        data["date"] = [time_stamp.date() for time_stamp in data["commence_time"]]

        # Order is kind of random, so we cannot trust that the provider
        # arranged home team first, then away in the teams list.
        data["home_team_index"] = [
            teams.index(home) for teams, home in zip(data["teams"], data["home_team"])
        ]
        data["away_team_index"] = 1 - data["home_team_index"]

        # Create a column for the away team.
        data["away_team"] = [
            teams[home_idx]
            for teams, home_idx in zip(data["teams"], data["away_team_index"])
        ]

        # Organize odds in smaller sub-samples.
        odds = [
            np.array([row["odds"]["h2h"] for row in sites]).mean(0)
            for sites in data["sites"]
        ]

        # Create odds columns.
        # The odds array has length equals to 3, and the index 1 is always the draw.
        data["home_team_odds"] = [
            row[i] for row, i in zip(odds, data["home_team_index"] * 2)
        ]
        data["draw_odds"] = [row[1] for row in odds]
        data["away_team_odds"] = [
            row[i] for row, i in zip(odds, data["away_team_index"] * 2)
        ]

        # Select columns to maintain.
        return data[
            [
                "date",
                "home_team",
                "away_team",
                "home_team_odds",
                "draw_odds",
                "away_team_odds",
            ]
        ]

    def betting_lines(self) -> pd.DataFrame:
        """ Get betting lines data frame. """
        # First check if the request wasn't already made to avoid excessive requests.
        cache_file_name = os.path.join(self.cache_folder, "betting_lines.json")
        if not os.path.exists(cache_file_name):

            # Create cache folder if doesn't exist yet.
            os.makedirs("cache", exist_ok=True)

            # Request.
            rqst = requests.get(
                url=self.host + "odds",
                verify=False,
                params={
                    "api_key": self.key,
                    "sport": "soccer_brazil_campeonato",
                    "region": "eu",
                    "mkt": "h2h",
                },
            ).json()["data"]

            # Save JSON to cache.
            with open(cache_file_name, "w") as file:
                json.dump(rqst, file)

        return self.clean_betting_lines(pd.read_json(cache_file_name))


def get_club_id(club_names: Sequence[str]) -> List[int]:
    """ Get club IDs from a sequence of names."""
    # Load club names mapping.
    with open(
        os.path.join(THIS_FOLDER, "data", "clubs_names.json"), encoding="utf-8"
    ) as file:
        names_mapping = json.load(file)["nome"]

    # Iterate through the sequence passe by the user.
    club_id = []
    for club in club_names:
        # iterate through the mapping.
        for i, names in names_mapping.items():
            # If the mapping is present in values, append the club ID.
            if club.lower() in [name.lower() for name in names]:
                club_id.append(i)
                break
        # If it iterated through all keys without finding it, append None.
        else:
            club_id.append(None)

    return club_id


def merge_clubs_and_odds(clubs: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """ Merge clubs and odds dataframes. """
    # Avoid in-place transformations
    clubs = clubs.copy()
    odds = odds.copy()

    # Transform names into IDs.
    odds["home_team"] = get_club_id(odds["home_team"])
    odds["away_team"] = get_club_id(odds["away_team"])

    # Create a new frame with the home team as index and and rename columns to merge.
    odds_home = odds.set_index("home_team", drop=True)
    odds_home = odds_home.rename(
        {"home_team_odds": "win_odds", "away_team_odds": "lose_odds",}, axis=1,
    )

    # Create a new frame with the away team as index and and rename columns to merge.
    odds_away = odds.set_index("away_team", drop=True)
    odds_away = odds_away.rename(
        {"away_team_odds": "win_odds", "home_team_odds": "lose_odds",}, axis=1,
    )

    # Merge home and way datasets.
    odds = odds_home.append(odds_away)
    # If a team has two games on the odds data frame, keep only the first.
    odds = odds.sort_values("date")[["win_odds", "draw_odds", "lose_odds"]]
    index = odds.index.drop_duplicates()
    odds = odds.loc[index]

    # Merge clubs and odds dataframes. Make sure both indexes are from the same type
    odds.index = odds.index.astype(int)
    clubs.index = clubs.index.astype(int)
    return pd.merge(clubs, odds, how="outer", left_index=True, right_index=True)


def get_clubs_with_odds(key: str, cache_folder: Optional[str] = None) -> pd.DataFrame:
    """ Get clubs data with odds included.. """
    # Get odds dataset.
    odds_api = TheOddsAPI(key=key, cache_folder=cache_folder)
    odds = odds_api.betting_lines()

    # Get clubs dataset.
    cartola_api = CartolaFCAPI()
    clubs = cartola_api.clubs()

    # Merge them.
    return merge_clubs_and_odds(clubs, odds)
