""" Athletes draft. """

import itertools
from typing import Iterator, Sequence

import numpy as np

import palpiteiro


def assign_captain(line_up: palpiteiro.LineUp) -> palpiteiro.LineUp:
    """ Assign the player with the most expected points as captain. """
    idx = np.argmax([player.predicted_points for player in line_up])
    line_up.captain = line_up[idx]
    return line_up


def draft(
    players: Sequence[palpiteiro.Player],
    schemes: Sequence[palpiteiro.Scheme],
    max_price: float,
) -> palpiteiro.LineUp:
    """ Draft best team possible using genetic algorithm. """
    for scheme in schemes:
        positions = []
        for pos, qty in scheme.dict.items():
            available_players = [player for player in players if player.position == pos]
            positions.append(tuple(itertools.combinations(available_players, qty)))

        teams = []
        for team in itertools.product(*positions):
            line_up = palpiteiro.LineUp(list(itertools.chain.from_iterable(team)))
            if line_up.price < max_price:
                teams.append(team)
