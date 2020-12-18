""" Athletes draft. """

import random
from typing import Sequence

import numpy as np

import palpiteiro


def assign_captain(line_up: palpiteiro.LineUp) -> palpiteiro.LineUp:
    """ Assign the player with the most expected points as captain. """
    idx = np.argmax([player.predicted_points for player in line_up])
    line_up.captain = line_up[idx]
    return line_up


def random_line_up(
    players: Sequence[palpiteiro.Player],
    schemes: Sequence[palpiteiro.Scheme],
    max_price: float,
) -> palpiteiro.LineUp:
    """ Create a random valid line-up. """
    # Select a random scheme.
    scheme = random.choice(schemes)

    # Separates players by position.
    players_dict = {
        pos: [player for player in players if player.position == pos]
        for pos in scheme.dict.keys()
    }

    # Create an empty line_up
    line_up = palpiteiro.LineUp([])

    # Iterate on positions to be draft on a random order.
    positions = [pos for pos, amount in scheme.dict.items() for _ in range(amount)]
    random.shuffle(positions)

    remaining_money = max_price
    for pos in positions:

        # Filter affordable players.
        affordable_players = [
            player
            for player in players_dict[pos]
            if player.price <= remaining_money and player not in line_up
        ]

        # If no affordable players, restart function.
        if len(affordable_players) == 0:
            return random_line_up(players=players, schemes=schemes, max_price=max_price)

        # Randomly choose a player and add to the line up.
        random_choice = random.choice(affordable_players)
        line_up.add(random_choice)

        # Remove price from money.
        remaining_money -= random_choice.price

    return assign_captain(line_up)


def available_players_generator(
    players: Sequence[palpiteiro.Player],
    line_up: palpiteiro.LineUp,
    max_player_price: float,
):
    """ Available players generator"""
    for player in players:
        if (player not in line_up.players) and (player.price < max_player_price):
            yield player


def mutate_line_up(
    line_up: palpiteiro.LineUp,
    players: Sequence[palpiteiro.Player],
    schemes: Sequence[palpiteiro.Scheme],
    max_price: float,
) -> palpiteiro.LineUp:
    """ Change a single random player in the line up. """
    # Avoid inplace transformations.
    line_up = line_up.copy()

    # Choose a random player to remove.
    player_to_remove = random.choice(line_up)
    line_up.remove(player_to_remove)

    positions = line_up.scheme.open_positions(schemes)

    # Estimate the maximum price that the new player can cost.
    max_player_price = max_price - line_up.price

    # Filter available players.
    available_players = [
        player
        for player in players
        if (player not in line_up.players)
        and (player.position in positions)
        and (player.price < max_player_price)
    ]

    if len(available_players) == 0:
        # If no available player. Apply recursion.
        return mutate_line_up(
            line_up=line_up, players=players, schemes=schemes, max_price=max_price,
        )

    new_player = random.choice(available_players)

    # Add new player.
    line_up.add(new_player)

    return assign_captain(line_up)


def crossover_line_up(
    line_up1: palpiteiro.LineUp, line_up2: palpiteiro.LineUp, max_price: float,
) -> palpiteiro.LineUp:
    """
    Cross-over two line_ups.

    Keeps line-up 1 scheme.
    """
    # Avoid inplace transformations.
    line_up1 = line_up1.copy()
    line_up2 = line_up2.copy()

    # Iterates through each player.
    for i in range(len(line_up1)):

        # Randomly decide to switch genes or not.
        if not random.choice([True, False]):
            # If false goes to the next player from line up 1.
            continue

        # If True, search for a player from the same position on Line Up 2.
        for j in range(len(line_up2)):
            if (
                line_up1[i].position == line_up2[j].position
                and line_up2[j] not in line_up1
                and line_up1[i] not in line_up2
            ):
                # Swap players and exit loop.
                line_up1[i], line_up2[j] = line_up2[j], line_up1[i]
                break

        # If no player from the same position is found. Nothing will happen.

    # If line up 1 price is affordable, return line up 1.
    if line_up1.price <= max_price:
        return assign_captain(line_up1)

    # If line up 1 is not affordable and line up 2 is, return line up 2.
    if line_up2.price <= max_price:
        return assign_captain(line_up2)

    # If price is too high, apply recursion.
    crossover_line_up(line_up1=line_up1, line_up2=line_up2, max_price=max_price)


def draft(
    individuals: int,
    generations: int,
    players: Sequence[palpiteiro.Player],
    schemes: Sequence[palpiteiro.Scheme],
    max_price: float,
    tournament_size: int,
    elite: int = 1,
) -> palpiteiro.LineUp:
    """ Draft best team possible using genetic algorithm. """
    # Create initial population.
    pop = [random_line_up(players, schemes, max_price) for _ in range(individuals)]

    # Run for the selected number of generations.
    for i in range(generations):

        # Rank entire population.
        pop = sorted(pop, key=lambda x: x.predicted_points, reverse=True)

        # Create new population.
        new_pop = []
        while len(new_pop) < individuals:

            # If elitism is activate:
            # Keep the best individual and do not create a new individual instead.
            if len(new_pop) < elite:
                new_pop.append(pop[len(new_pop) + 1])
                continue

            # If elite was already separated, begin tournaments.
            # Rank randomly selected individuals and rank them by fitness.
            ranking = sorted(
                random.sample(pop, tournament_size),
                key=lambda x: x.predicted_points,
                reverse=True,
            )

            # Coin-flip. If True crossover, else mutation.
            if random.choice([True, False]):
                offspring = crossover_line_up(
                    line_up1=ranking[0], line_up2=ranking[1], max_price=max_price
                )
            else:
                offspring = mutate_line_up(
                    line_up=ranking[0],
                    players=players,
                    schemes=schemes,
                    max_price=max_price,
                )

            new_pop.append(offspring)
        pop = new_pop

    # Return the line up with the most predicted points.
    return sorted(pop, key=lambda x: x.predicted_points, reverse=True)[0]
