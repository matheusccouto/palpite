""" palpiteiro.data unit-tests. """

import os

import pandas as pd

import palpiteiro.data

THIS_FOLDER = os.path.dirname(__file__)


class TestTheOddsAPI:
    """ Test TheOddsAPI class. """

    def test_load_from_cache(self):
        """ Test loading data from cache. """
        # Load using API
        odds_api = palpiteiro.data.TheOddsAPI("1902")  # Fake key.
        loaded = odds_api.betting_lines()

        # Load directly from cache.
        cache = pd.read_json(
            os.path.join(THIS_FOLDER, "cache", "betting_lines.json")
        )
        cache = odds_api.clean_betting_lines(cache)

        pd.testing.assert_frame_equal(loaded, cache)


class TestClubsAndOddsMerge:
    """ Test functions related to merging odds to the clubs dataframe. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """
        cartola_api = palpiteiro.data.CartolaFCAPI()
        cls.clubs = cartola_api.clubs()

        cls.odds = pd.read_csv(
            os.path.join(THIS_FOLDER, "data", "odds_api", "odds.csv"), index_col=0
        )

    def test_get_club_id(self):
        """ Test get_club_id function. """
        club_id = palpiteiro.data.get_club_id(
            [
                "Cuiaba",
                "sao paulo",
                "Bragantino-SP",
                "atletico Mineiro",
                "Gremio",
                "vitoria",
                "Criciuma",
                "parana",
                "Goias",
                "Atletico paranaense",
                "Avai",
                "aMeRiCa MiNeIrO",
                "Nautico",
                "America-RN",
                "Atletico Goianiense",
            ]
        )
        assert None not in club_id

    def test_merge_clubs_and_odds_exists(self):
        """ Test function merge_clubs_and_odds on teams that have odds. """
        merged = palpiteiro.data.merge_clubs_and_odds(clubs=self.clubs, odds=self.odds)
        assert merged.loc[266]["nome"] == "Fluminense"
        assert round(merged.loc[266]["win_odds"], 2) == 3.26
        assert round(merged.loc[266]["draw_odds"], 2) == 2.10
        assert round(merged.loc[266]["lose_odds"], 2) == 3.43

    def test_merge_clubs_and_odds_nan(self):
        """ Test function merge_clubs_and_odds on teams that do not have odds. """
        merged = palpiteiro.data.merge_clubs_and_odds(clubs=self.clubs, odds=self.odds)
        assert merged.loc[364]["nome"] == "Remo"
        assert pd.isna(merged.loc[364]["win_odds"])
        assert pd.isna(merged.loc[364]["draw_odds"])
        assert pd.isna(merged.loc[364]["lose_odds"])

    def test_get_club_data(self):
        """ Test function get_clubs """
        # This is a fake key, but there is no problem
        # because it will use the cache folder.
        clubs = palpiteiro.data.get_clubs_with_odds(
            key="1902", cache_folder=os.path.join(THIS_FOLDER, "cache")
        )
        assert clubs.loc[266]["nome"] == "Fluminense"
        assert round(clubs.loc[266]["win_odds"], 2) == 3.26
        assert round(clubs.loc[266]["draw_odds"], 2) == 2.10
        assert round(clubs.loc[266]["lose_odds"], 2) == 3.43
        assert clubs.loc[364]["nome"] == "Remo"
        assert pd.isna(clubs.loc[364]["win_odds"])
        assert pd.isna(clubs.loc[364]["draw_odds"])
        assert pd.isna(clubs.loc[364]["lose_odds"])
