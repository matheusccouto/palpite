""" Cartola FC tips. """

import os
from typing import Sequence, Optional, List, Dict

import joblib
import pandas as pd

THIS_FOLDER = os.path.dirname(__file__)
MODEL_PATH = os.path.join(THIS_FOLDER, "data", "model.pkl")
MODEL = joblib.load(MODEL_PATH)


class Club:
    """
    Brasileirão Série A club.

    Uses Cartola FC API IDs.
    """

    def __init__(self, club_id: int, clubs: pd.DataFrame):
        self.id = club_id
        self.clubs = clubs

    def __eq__(self, other: "Club") -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"

    @property
    def name(self) -> str:
        """ Club name. """
        return self.clubs.loc[self.id]["nome"]

    @property
    def abbreviation(self) -> str:
        """ Club abbreviation. """
        return self.clubs.loc[self.id]["abreviacao"]

    @property
    def logo(self) -> str:
        """ Club logo. """
        return self.clubs.loc[self.id]["escudos"]["60x60"]

    @property
    def win_odds(self) -> float:
        """ Odds of winning in the next match. """
        return self.clubs.loc[self.id]["win_odds"]

    @property
    def draw_odds(self) -> float:
        """ Odds of drawing in the next match. """
        return self.clubs.loc[self.id]["draw_odds"]

    @property
    def lose_odds(self) -> float:
        """ Odds of losing in the next match. """
        return self.clubs.loc[self.id]["lose_odds"]


class Player:
    """ Cartola FC player. """

    # Mapping from Cartola FC API position to machine learning model position.
    position_map = {
        1: 1,  # Goalkeeper
        2: 3,  # Fullback
        3: 2,  # Defender
        4: 4,  # Midfielder
        5: 5,  # Forward
        6: 0,  # Coach
    }

    # Status map
    status_map = {
        7: 4,  # Expected
        5: 1,  # Injured
        2: 3,  # Doubt
        3: 0,  # Suspended
        6: 2,  # Null
    }

    def __init__(
        self, player_id: int, players: pd.DataFrame, clubs: pd.DataFrame,
    ):
        self.id = player_id
        self.players = players
        self.clubs = clubs

    def __eq__(self, other: "Player") -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"

    @property
    def name(self) -> str:
        """ Player name. """
        return self.players.loc[self.id]["apelido"]

    @property
    def club(self) -> Club:
        """ Player club. """
        return Club(club_id=self.players.loc[self.id]["clube_id"], clubs=self.clubs,)

    @property
    def photo(self) -> str:
        """ Player photo. """
        return self.players.loc[self.id]["foto"]

    @property
    def position(self) -> int:
        """ Player position. """
        return self.players.loc[self.id]["posicao_id"]

    @property
    def status(self) -> int:
        """ Player status. """
        return self.players.loc[self.id]["status_id"]

    @property
    def matches(self) -> int:
        """ Player amount of played matches. """
        return self.players.loc[self.id]["jogos_num"]

    @property
    def points(self) -> float:
        """ Player points on club's last match. """
        return self.players.loc[self.id]["pontos_num"]

    @property
    def mean(self) -> float:
        """ Player mean points considering matches he has played. """
        return self.players.loc[self.id]["media_num"]

    @property
    def price(self) -> float:
        """ Player price. """
        return self.players.loc[self.id]["preco_num"]

    @property
    def variation(self) -> float:
        """ Player price variation. """
        return self.players.loc[self.id]["variacao_num"]

    @property
    def scouts(self) -> Dict[str, int]:
        """ Player scouts. """
        return self.players.loc[self.id]["scout"]

    @property
    def win_odds(self) -> float:
        """ Get odds of his club winning in the next match. """
        return self.club.win_odds

    @property
    def draw_odds(self) -> float:
        """ Get odds of his club drawing in the next match. """
        return self.club.draw_odds

    @property
    def lose_odds(self) -> float:
        """ Get odds of his club losing in the next match. """
        return self.club.lose_odds

    @property
    def predicted_points(self) -> float:
        """ Get predicted points using a machine learning model. """
        status = self.status_map[self.status]
        # If the player is suspended, injured or null,
        # it is expected to score no points at all.
        if status in [0, 1, 2]:
            return 0.0

        # Can't make predictions if there aren't odds on the player's club.
        if any(
            [pd.isna(self.win_odds), pd.isna(self.lose_odds), pd.isna(self.draw_odds)]
        ):
            raise ValueError(
                f"Can't make predictions for {self} "
                f"because {self.club} does not have available odds."
            )

        return MODEL.predict(
            [
                [
                    self.position_map[self.position],
                    status,
                    self.matches,
                    self.mean,
                    self.price,
                    self.variation,
                    self.win_odds,
                    self.lose_odds,
                    self.draw_odds,
                ]
            ]
        )[0][0]


def create_all_players(players: pd.DataFrame, clubs: pd.DataFrame) -> List[Player]:
    """ Create all players from a players dataframe. """
    return [Player(i, players, clubs) for i in players.index]


class Scheme:
    """ Cartola FC schemes. """

    # Cartola FC ID to Position:
    # 1 - Goalkeeper
    # 2 - Fullback
    # 3 - Defender
    # 4 - Midfielder
    # 5 - Forward
    # 6 - Coach

    def __init__(
        self,
        goalkeepers: int,
        fullbacks: int,
        defenders: int,
        midfielders: int,
        forwards: int,
        coaches: int,
    ):
        self.goalkeepers = goalkeepers
        self.fullbacks = fullbacks
        self.defenders = defenders
        self.midfielders = midfielders
        self.forwards = forwards
        self.coaches = coaches

    def __eq__(self, other: "Scheme") -> bool:
        return self.dict == other.dict

    def __str__(self) -> str:
        return f"{self.defenders + self.fullbacks}-{self.midfielders}-{self.forwards}"

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"

    @property
    def dict(self) -> Dict[int, int]:
        """ Get scheme dict where  keys are position id and values the amount. """
        return {
            1: self.goalkeepers,
            2: self.fullbacks,
            3: self.defenders,
            4: self.midfielders,
            5: self.forwards,
            6: self.coaches,
        }


def create_schemes(schemes: pd.DataFrame) -> List[Scheme]:
    """ Create valid schemes list. """
    scheme_list = list()
    for _, scheme in schemes.iterrows():
        scheme_list.append(
            Scheme(
                goalkeepers=scheme["posicoes"]["gol"],
                fullbacks=scheme["posicoes"]["lat"],
                defenders=scheme["posicoes"]["zag"],
                midfielders=scheme["posicoes"]["mei"],
                forwards=scheme["posicoes"]["ata"],
                coaches=scheme["posicoes"]["tec"],
            )
        )
    return scheme_list


class LineUp:
    """ Cartola FC team line-up. """

    # Cartola FC ID to Position:
    # 1 - Goalkeeper
    # 2 - Fullback
    # 3 - Defender
    # 4 - Midfielder
    # 5 - Forward
    # 6 - Coach

    def __init__(self, players: Sequence[Player]):
        self.players = players
        self._captain: Optional[Player] = None

    def __eq__(self, other: "LineUp") -> bool:
        these_players = sorted(self.players, key=lambda x: x.id)
        other_players = sorted(other.players, key=lambda x: x.id)
        return these_players == other_players

    def is_valid(self, schemes: Sequence[Scheme]):
        """ Checks if the line-up is valid. """
        return any([self.scheme == valid for valid in schemes])

    @property
    def goalkeepers(self) -> Sequence[Player]:
        """ Get goalkeepers in the line up. """
        return [player for player in self.players if player.position == 1]

    @property
    def fullbacks(self) -> Sequence[Player]:
        """ Get fullbacks in the line up. """
        return [player for player in self.players if player.position == 2]

    @property
    def defenders(self) -> Sequence[Player]:
        """ Get defenders in the line up. """
        return [player for player in self.players if player.position == 3]

    @property
    def midfielders(self) -> Sequence[Player]:
        """ Get midfielders in the line up. """
        return [player for player in self.players if player.position == 4]

    @property
    def forwards(self) -> Sequence[Player]:
        """ Get forwards in the line up. """
        return [player for player in self.players if player.position == 5]

    @property
    def coaches(self) -> Sequence[Player]:
        """ Get coaches in the line up. """
        return [player for player in self.players if player.position == 6]

    @property
    def captain(self) -> Player:
        """ Get and set captain. """
        if self._captain is None:
            raise ValueError("This line up does not have a captain yet.")
        return self._captain

    @captain.setter
    def captain(self, value: Player):
        if value not in self.players:
            raise ValueError("The captain must be one of the players from the line-up")
        self._captain = value

    @property
    def scheme(self) -> Scheme:
        """ Get scheme. """
        return Scheme(
            goalkeepers=len(self.goalkeepers),
            fullbacks=len(self.fullbacks),
            defenders=len(self.defenders),
            midfielders=len(self.midfielders),
            forwards=len(self.forwards),
            coaches=len(self.coaches),
        )

    @property
    def points(self) -> int:
        """ Get line up points. """
        return sum(
            [
                2 * player.points if player == self.captain else player.points
                for player in self.players
            ]
        )

    @property
    def predicted_points(self) -> int:
        """ Get line up points. """
        return sum(
            [
                2 * player.predicted_points
                if player == self.captain
                else player.predicted_points
                for player in self.players
            ]
        )
