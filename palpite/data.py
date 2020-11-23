""" Data requesting and wrangling. """

import datetime
import os
from typing import Optional

import json
import numpy as np
import pandas as pd
import requests


def request_to_df(url: str, key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """ Create dataframe from request data. """
    data = requests.get(url, verify=False, **kwargs).json()

    # If specified some inner key.
    if key is not None:
        data = data[key]

    # If it is a dictionary, use the values method.
    if isinstance(data, dict):
        data = data.values()

    data_frame = pd.DataFrame()
    for row in data:
        data_frame = data_frame.append(row, ignore_index=True)
    return data_frame


class CartolaFC:
    """ A high level wrapper for the Cartola FC API. """

    host = r"https://api.cartolafc.globo.com/"

    def clubs(self) -> pd.DataFrame:
        """ Get clubs data frame. """
        return request_to_df(self.host + r"atletas/mercado", "clubes")

    def matches(self) -> pd.DataFrame:
        """ Get next matches data frame. """
        return request_to_df(self.host + r"partidas", "partidas")

    def players(self) -> pd.DataFrame:
        """ Get players data frame. """
        return request_to_df(self.host + r"atletas/mercado", "atletas")

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

    def __init__(self, key: str):
        self.key = key

    @staticmethod
    def clean_betting_lines(data: pd.DataFrame) -> pd.DataFrame:
        """ Clean betting lines dataframe. """
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
        cache_file_name = os.path.join("cache", "betting_lines.json")
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


class FootballData:
    """ A football-data.co.uk data wrangler. """

    @staticmethod
    def _format_date(series: pd.Series) -> pd.Series:
        """ Format date series. """
        return pd.Series(
            [
                datetime.date(*[int(val) for val in date.split("/")[::-1]])
                for date in series
            ]
        )

    def historical_data(self, data: pd.DataFrame):
        """ Clean historic data. """

        new = pd.DataFrame()

        new["date"] = self._format_date(data["Date"])
        new["home_team"] = data["Home"]
        new["away_team"] = data["Away"]
        new["home_team_odds"] = data["AvgH"]
        new["draw_odds"] = data["AvgD"]
        new["away_team_odds"] = data["AvgA"]

        return new
