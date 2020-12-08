""" Unit-tests for palpiteiro package. """

import os

import palpiteiro
import palpiteiro.data

THIS_FOLDER = os.path.dirname(__file__)


class TestClub:
    """ Unit-tests for class Club. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """
        cls.clubs = palpiteiro.data.get_clubs_with_odds(
            key="1902", cache_folder=os.path.join(THIS_FOLDER, "cache")
        )  # Fake key. But doesn't matter.
        cls.club = palpiteiro.Club(266, clubs=cls.clubs)

    def test_name(self):
        """ Test team name. """
        assert self.club.name == self.clubs.loc[266]["nome"]

    def test_abbreviation(self):
        """ Test team abbreviation. """
        assert self.club.abbreviation == self.clubs.loc[266]["abreviacao"]

    def test_logo(self):
        """ Test team logos. """
        assert self.club.logo == self.clubs.loc[266]["escudos"]["60x60"]

    def test_win_odds(self):
        """ Test team winning odds. """
        assert self.club.win_odds == self.clubs.loc[266]["win_odds"]

    def test_draw_odds(self):
        """ Test team drawing odds. """
        assert self.club.draw_odds == self.clubs.loc[266]["draw_odds"]

    def test_lose_odds(self):
        """ Test team losing odds. """
        assert self.club.lose_odds == self.clubs.loc[266]["lose_odds"]

    def test_eq(self):
        """ Test equivalency. """
        assert palpiteiro.Club(266, self.clubs) == palpiteiro.Club(266, self.clubs)

    def test_not_eq(self):
        """ Test unequivalency. """
        assert not palpiteiro.Club(266, self.clubs) == palpiteiro.Club(267, self.clubs)


class TestAthlete:
    """ Unit-tests for athlete class. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """
        # Get clubs dataset.
        cls.clubs = palpiteiro.data.get_clubs_with_odds(
            key="1902", cache_folder=os.path.join(THIS_FOLDER, "cache")
        )  # Fake key. But doesn't matter.

        # Get players.
        cartola_api = palpiteiro.data.CartolaFCAPI()
        cls.players = cartola_api.players()

        # Create a player.
        cls.player = palpiteiro.Player(38162, players=cls.players, clubs=cls.clubs)

    def test_name(self):
        """ Test player name. """
        return self.player.name == "Fred"

    def test_photo(self):
        """ Test player photo. """
        return (
            self.player.photo
            == r"https://s.glbimg.com/es/sde/f/2020/08/11/1546fc5edcfb41d2e8f79f6eca1f6899_FORMATO.png"
        )

    def test_club(self):
        """ Test player club id. """
        return self.player.club.id == 266

    def test_position(self):
        """ Test player position. """
        return self.player.position == 5

    def test_status(self):
        """ Test player status. """
        return self.player.status == 5

    def test_points(self):
        """ Test player points on club's last match. """
        return self.player.points == 0

    def test_mean(self):
        """ Test player mean points on matches he has played.. """
        return self.player.mean == 2.1

    def test_price(self):
        """ Test player price. """
        return self.player.price == 3.4

    def test_variation(self):
        """ Test player price variation since last match. """
        return self.player.variation == 0

    def test_scouts(self):
        """ Test getting player scouts. """
        return [
            scout in self.player.scouts
            for scout in ["A", "CA", "DS", "FC", "FD", "FF", "FS", "G", "I", "PI"]
        ]

    def test_eq(self):
        """ Test equivalency. """
        player_1 = palpiteiro.Player(38162, self.players, self.clubs)
        player_2 = palpiteiro.Player(38162, self.players, self.clubs)
        assert player_1 == player_2

    def test_not_eq(self):
        """ Test unequivalency. """
        player_1 = palpiteiro.Player(38162, self.players, self.clubs)
        player_2 = palpiteiro.Player(38913, self.players, self.clubs)
        assert not player_1 == player_2

    def test_predicted_points_null(self):
        """ Test if the machine learning makes a null prediction for when injured. """
        assert self.player.predicted_points == 0

    def test_predicted_points(self):
        """ Test if the machine learning makes prediction. """
        player = palpiteiro.Player(38913, self.players, self.clubs)
        assert player.predicted_points > 0


class TestScheme:
    """ Unit tests for Scheme class. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """
        cls.scheme = palpiteiro.Scheme(
            goalkeepers=1,
            fullbacks=2,
            defenders=2,
            midfielders=4,
            forwards=2,
            coaches=1,
        )

    def test_each_position(self):
        """ Test each position property. """
        assert self.scheme.goalkeepers == 1
        assert self.scheme.fullbacks == 2
        assert self.scheme.defenders == 2
        assert self.scheme.midfielders == 4
        assert self.scheme.forwards == 2
        assert self.scheme.coaches == 1

    def test_str(self):
        """ Test string transformation. """
        assert str(self.scheme) == "4-4-2"

    def test_dict(self):
        """ test dict property. """
        assert self.scheme.dict == {
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 2,
            6: 1,
        }

    @staticmethod
    def test_create_schemes():
        """ Test create scheme function. """
        cartola_api = palpiteiro.data.CartolaFCAPI()
        raw_schemes = cartola_api.schemes()
        schemes = palpiteiro.create_schemes(raw_schemes)

        expected_schemes = [
            "3-4-3",
            "3-5-2",
            "4-3-3",
            "4-4-2",
            "4-5-1",
            "5-3-2",
            "5-4-1",
        ]

        assert [str(scheme) in expected_schemes for scheme in schemes]


class TestLineUp:
    """ Unit-tests for LineUp class. """

    @classmethod
    def setup_class(cls):
        """ Setup class. """

        # Get clubs
        clubs = palpiteiro.data.get_clubs_with_odds(
            key="1902", cache_folder=os.path.join(THIS_FOLDER, "cache")
        )  # Fake key.

        # Get players
        cartola_api = palpiteiro.data.CartolaFCAPI()
        players = cartola_api.players()

        # Get schemes
        cls.schemes = palpiteiro.create_schemes(cartola_api.schemes())

        # Get all players.
        players = palpiteiro.create_all_players(players, clubs)
        # Filter team
        players = [player for player in players if player.club.id == 266]

        cls.goalkeepers = [player for player in players if player.position == 1]
        cls.fullbacks = [player for player in players if player.position == 2]
        cls.defenders = [player for player in players if player.position == 3]
        cls.midfielders = [player for player in players if player.position == 4]
        cls.forwards = [player for player in players if player.position == 5]
        cls.coaches = [player for player in players if player.position == 6]

        cls.line_up_list = [
            cls.goalkeepers[0],
            cls.fullbacks[0],
            cls.fullbacks[1],
            cls.defenders[0],
            cls.defenders[1],
            cls.midfielders[0],
            cls.midfielders[1],
            cls.midfielders[2],
            cls.midfielders[3],
            cls.forwards[0],
            cls.forwards[1],
            cls.coaches[0],
        ]

    def test_invalid_line_up1(self):
        """ Test setting an invalid line-up. """
        assert not palpiteiro.LineUp([self.goalkeepers[0]]).is_valid(self.schemes)

    def test_invalid_line_up2(self):
        """ Test setting an invalid line-up. """
        invalid_line_up = self.line_up_list.copy()
        invalid_line_up[0] = self.fullbacks[-1]
        assert not palpiteiro.LineUp(invalid_line_up).is_valid(self.schemes)

    def test_invalid_line_up3(self):
        """ Test setting an invalid line-up. """
        assert not palpiteiro.LineUp(self.line_up_list + self.coaches[-1:]).is_valid(
            self.schemes
        )

    def test_valid_line_up(self):
        """ Test setting a valid line-up. """
        assert palpiteiro.LineUp(self.line_up_list).is_valid(self.schemes)

    def test_get_scheme(self):
        """ Test line_up scheme. """
        my_scheme = palpiteiro.LineUp(self.line_up_list).scheme
        ref_scheme = palpiteiro.Scheme(
            goalkeepers=1,
            fullbacks=2,
            defenders=2,
            midfielders=4,
            forwards=2,
            coaches=1,
        )
        assert my_scheme == ref_scheme

    def test_captain(self):
        """ Test set captain. """
        line_up = palpiteiro.LineUp(self.line_up_list)
        line_up.captain = line_up.players[0]
        assert line_up.captain is self.line_up_list[0]

    def test_positions(self):
        """ Test each position. """
        line_up = palpiteiro.LineUp(self.line_up_list)
        assert line_up.goalkeepers == self.goalkeepers[:1]
        assert line_up.fullbacks == self.fullbacks[:2]
        assert line_up.defenders == self.defenders[:2]
        assert line_up.midfielders == self.midfielders[:4]
        assert line_up.forwards == self.forwards[:2]
        assert line_up.coaches == self.coaches[:1]

    def test_points(self):
        """ Test get current points. """
        line_up = palpiteiro.LineUp(self.line_up_list)
        line_up.captain = line_up.players[0]
        assert line_up.points > 0

    def test_predicted_points(self):
        """ Test get current points. """
        line_up = palpiteiro.LineUp(self.line_up_list)
        line_up.captain = line_up.players[0]
        assert line_up.predicted_points > 0
