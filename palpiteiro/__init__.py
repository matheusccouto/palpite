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

    def __init__(self, club_id: int, clubs: dict):
        self.id = club_id
        self.clubs = clubs
        # Transform in dict to improve performance.
        self._dict = self.clubs[self.id]

    def __eq__(self, other: "Club") -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"

    @property
    def name(self) -> str:
        """ Club name. """
        return self._dict["nome"]

    @property
    def abbreviation(self) -> str:
        """ Club abbreviation. """
        return self._dict["abreviacao"]

    @property
    def logo(self) -> str:
        """ Club logo. """
        return self._dict["escudos"]["45x45"]

    @property
    def win_odds(self) -> float:
        """ Odds of winning in the next match. """
        return self._dict["win_odds"]

    @property
    def draw_odds(self) -> float:
        """ Odds of drawing in the next match. """
        return self._dict["draw_odds"]

    @property
    def lose_odds(self) -> float:
        """ Odds of losing in the next match. """
        return self._dict["lose_odds"]


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

    position_abbreviation_map = {
        1: "GOL",
        2: "LAT",
        3: "ZAG",
        4: "MEI",
        5: "ATA",
        6: "TEC",
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
        self, player_id: int, players: dict, clubs: dict,
    ):
        self.id = player_id
        self.players = players
        self.clubs = clubs

        self._dict = players[self.id]

        # Get player club.
        self.club = Club(club_id=self._dict["clube_id"], clubs=self.clubs)

        self.predicted_points = 0
        self.update_predicted_points()

    def __eq__(self, other: "Player") -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"

    @property
    def name(self) -> str:
        """ Player name. """
        return self._dict["apelido"]

    @property
    def photo(self) -> str:
        """ Player photo. """
        return self._dict["foto"].replace("_FORMATO", "_140x140")

    @property
    def position(self) -> int:
        """ Player position. """
        return self._dict["posicao_id"]

    @property
    def position_abbreviation(self) -> str:
        """ Player position name abbreviation"""
        return self.position_abbreviation_map[self.position]

    @property
    def status(self) -> int:
        """ Player status. """
        return self._dict["status_id"]

    @property
    def matches(self) -> int:
        """ Player amount of played matches. """
        return self._dict["jogos_num"]

    @property
    def points(self) -> float:
        """ Player points on club's last match. """
        return self._dict["pontos_num"]

    @property
    def mean(self) -> float:
        """ Player mean points considering matches he has played. """
        return self._dict["media_num"]

    @property
    def price(self) -> float:
        """ Player price. """
        return self._dict["preco_num"]

    @property
    def variation(self) -> float:
        """ Player price variation. """
        return self._dict["variacao_num"]

    @property
    def scouts(self) -> Dict[str, int]:
        """ Player scouts. """
        return self._dict["scout"]

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

    def update_predicted_points(self) -> float:
        """ Estimate predicted points using a machine learning model. """
        status = self.status_map[self.status]
        # If the player is suspended, injured or null,
        # it is expected to score no points at all.
        if status in [0, 1, 2]:
            return 0.0

        # Can't make predictions if there aren't odds on the player's club.
        if any(
            [pd.isna(self.win_odds), pd.isna(self.lose_odds), pd.isna(self.draw_odds)]
        ):
            return 0.0
            # raise ValueError(
            #     f"Can't make predictions for {self} "
            #     f"because {self.club} does not have available odds."
            # )

        self.predicted_points = MODEL.predict(
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

    @property
    def is_predictable(self):
        """ Check if it is possible to make the predictions. """
        return pd.notna(self.win_odds)


def create_all_players(players: pd.DataFrame, clubs: pd.DataFrame) -> List[Player]:
    """ Create all players from a players dataframe. """
    players_dict = players.to_dict(orient="index")
    clubs_dict = clubs.to_dict(orient="index")
    return [Player(i, players_dict, clubs_dict) for i in players.index]


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

    def is_position_open(self, position: int, schemes: Sequence["Scheme"]) -> bool:
        """ Check if a position can be filled. """
        scheme_dict = self.dict.copy()
        scheme_dict[position] += 1
        return any([scheme_dict == other.dict for other in schemes])

    def open_positions(self, schemes: Sequence["Scheme"]) -> List[int]:
        """ Get positions that can be filled. """
        # Avoid inplace
        return [pos for pos in self.dict.keys() if self.is_position_open(pos, schemes)]


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
        self.players = list(players)
        self._captain: Optional[Player] = None

    def __eq__(self, other: "LineUp") -> bool:
        these_players = sorted(self.players, key=lambda x: x.id)
        other_players = sorted(other.players, key=lambda x: x.id)
        return these_players == other_players

    def __contains__(self, item: Player) -> bool:
        return item in self.players

    def __getitem__(self, key: int) -> Player:
        return self.players[key]

    def __setitem__(self, key: int, value: Player) -> None:
        self.players[key] = value

    def __len__(self) -> int:
        return len(self.players)

    def __iter__(self) -> Player:
        yield from self.players

    def __str__(self) -> str:
        players_list = [
            str(player) for player in sorted(self.players, key=lambda x: x.position)
        ]
        return f"LineUp {self.scheme} {', '.join(players_list)}"

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"

    def add(self, player: Player) -> None:
        """ Add player to the line up. """
        self.players.append(player)

    def remove(self, player: Player) -> None:
        """ Remove player from the line up. """
        self.players.remove(player)

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
    def price(self) -> float:
        """ Get line up price. """
        return sum([player.price for player in self.players])

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

    def copy(self):
        """ Copy to a new instance. """
        return LineUp(self.players)

    @property
    def dataframe(self):
        """ Export to a pandas DataFrame instance. """
        data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Club Photo": player.club.logo,
                        "Club": player.club.name,
                        "Photo": player.photo,
                        "Name": f"{player.name} (C)"
                        if player == self.captain
                        else player.name,
                        "Position": player.position,
                        "Position Name": player.position_abbreviation,
                        "Predicted Points": player.predicted_points,
                    },
                    index=[0],
                )
                for player in self
            ],
            ignore_index=True,
        )
        return data.sort_values("Position", ignore_index=True)