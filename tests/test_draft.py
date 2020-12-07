""" Unit-tests for palpite.draft """

import time
import os

import pandas as pd
import pytest

import palpite
import palpite.draft
import palpite.data

THIS_FOLDER = os.path.dirname(__file__)


# Get clubs.
clubs = palpite.data.get_clubs_with_odds("1902", os.path.join(THIS_FOLDER, "cache"))

# Initialize Cartola FC API.
cartola_fc_api = palpite.data.CartolaFCAPI()

# Players.
players = palpite.create_all_players(cartola_fc_api.players(), clubs)
players = [player for player in players if player.status in [2, 7]]
players = [player for player in players if pd.notna(player.club.win_odds)]

# Schemes.
schemes = palpite.create_schemes(cartola_fc_api.schemes())


class TestRandomLineUp:
    """ Test random_line_up function."""

    def test_is_valid(self):
        """ Test if generated line up is valid. """
        line_up = palpite.draft.random_line_up(players, schemes, 1e6)
        assert line_up.is_valid(schemes)

    def test_is_expensive(self):
        """
        Test if it raises an error when it is impossible to create a team with the
        available money.
        """
        with pytest.raises(RecursionError):
            palpite.draft.random_line_up(players, schemes, 0)

    def test_affordable(self):
        """ Make sure all line ups generated are below max price."""
        prices = [palpite.draft.random_line_up(players, schemes, 70).price for _ in range(100)]
        assert max(prices) <= 70

    def test_perfomance(self):
        """ Test if it runs functions 100 times in less than a second. """
        start = time.time()
        for _ in range(100):
            palpite.draft.random_line_up(players, schemes, 1e6)
        end = time.time()
        assert end - start < 1  # seconds


class TestMutateLineUp:
    """ Unit tests for mutate_line_up function. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """
        cls.line_up = palpite.draft.random_line_up(
            players=players, schemes=schemes, max_price=1e6
        )

    def test_not_equal(self):
        """ Check that the mutated line up is not equal. """
        new_line_up = palpite.draft.mutate_line_up(
            line_up=self.line_up, players=players, schemes=schemes, max_price=1e6,
        )
        assert new_line_up != self.line_up

    def test_perfomance(self):
        """ Test if it runs functions 100 times in less than a second. """
        start = time.time()
        for _ in range(100):
            palpite.draft.mutate_line_up(self.line_up, players, schemes, 1e6)
        end = time.time()
        assert end - start < 1  # seconds


class TestCrossoverLineUp:
    """ Unit tests for crossover_line_up function. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """
        cls.line_up1 = palpite.draft.random_line_up(
            players=players, schemes=schemes, max_price=1e6
        )
        cls.line_up2 = palpite.draft.random_line_up(
            players=players, schemes=schemes, max_price=1e6
        )

    def test_perfomance(self):
        """ Test if it runs functions 100 times in less than a second. """
        start = time.time()
        for _ in range(100):
            palpite.draft.crossover_line_up(
                line_up1=self.line_up1, line_up2=self.line_up2, max_price=1e6
            )
        end = time.time()
        assert end - start < 1  # seconds


class TestDraft:
    """ Unit tests for draft class. """

    def test_duplicates(self):
        """ Make sure there aren't duplicates on the final team. """
        best_line_up = palpite.draft.draft(
            individuals=50,
            generations=10,
            players=players,
            schemes=schemes,
            max_price=1e6,
            tournament_size=5,
        )
        players_ids = [player.id for player in best_line_up]
        assert len(players_ids) == len(set(players_ids))

    def test_draft(self):
        """ Test main functionality. """
        best_line_up = palpite.draft.draft(
            individuals=50,
            generations=10,
            players=players,
            schemes=schemes,
            max_price=100,
            tournament_size=5,
        )
        assert best_line_up.points > 0
