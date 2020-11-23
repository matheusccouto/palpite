""" palpite.data unit-tests. """

import datetime
import os

import pandas as pd

from palpite import data

THIS_FOLDER = os.path.dirname(__file__)


class TestFootballData:
    """ Test football-data.co.uk data wrangler. """

    dataset_path = os.path.join(
        THIS_FOLDER, "data", "football_data_co_uk", "historical_betting_lines.csv"
    )

    @classmethod
    def setup_class(cls):
        """ Setup tests. """
        dataset = pd.read_csv(cls.dataset_path)
        football_data = data.FootballData()
        cls.row = football_data.historical_data(dataset).loc[0]

    def test_date(self):
        """ Test date column. """
        assert self.row["date"] == datetime.date(year=2012, month=11, day=11)

    def test_home_team(self):
        """ Test home team column. """
        assert self.row["home_team"] == "Palmeiras"

    def test_away_team(self):
        """ Test away team column. """
        assert self.row["away_team"] == "Fluminense"

    def test_home_team_odds(self):
        """ Test home team odds column. """
        assert self.row["home_team_odds"] == 2.44

    def test_away_team_odds(self):
        """ Test home team odds column. """
        assert self.row["away_team_odds"] == 2.7

    def test_draw_odds(self):
        """ Test draw odds column. """
        assert self.row["draw_odds"] == 3.3
