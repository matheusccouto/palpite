from typing import Optional

import pandas as pd
import requests
from soccerapi.api import Api888Sport


def request_to_df(url: str, key: Optional[str] = None) -> pd.DataFrame:
    """ Create dataframe from request data. """
    data = requests.get(url, verify=False).json()

    # If specified some inner key.
    if key is not None:
        data = data[key]

    # If it is a dictionary, use the values method.
    if isinstance(data, dict):
        data = data.values()

    df = pd.DataFrame()
    for row in data:
        df = df.append(row, ignore_index=True)
    return df


def clubs() -> pd.DataFrame:
    """ Get clubs data. """
    return request_to_df(r"https://api.cartolafc.globo.com/atletas/mercado", "clubes")


def matches() -> pd.DataFrame:
    """ Get next matches data. """
    return request_to_df(r"https://api.cartolafc.globo.com/partidas", key="partidas")


def players() -> pd.DataFrame:
    """ Get players data. """
    return request_to_df(r"https://api.cartolafc.globo.com/atletas/mercado", "atletas")


def schemes():
    """ Get schemes data. """
    return request_to_df(r"https://api.cartolafc.globo.com/esquemas")


def positions() -> pd.DataFrame:
    """ Get positions data. """
    return request_to_df(r"https://api.cartolafc.globo.com/atletas/mercado", "posicoes")


def status() -> pd.DataFrame:
    """ Get status data. """
    return request_to_df(r"https://api.cartolafc.globo.com/atletas/mercado", "status")


def bet_lines() -> pd.DataFrame:
    """ Get bet lines data. """
    api = Api888Sport()
    url = 'https://www.888sport.com/#/filter/football/italy/serie_a'
    odds = api.odds(url)
    return odds


if __name__ == "__main__":

    print(bet_lines())
