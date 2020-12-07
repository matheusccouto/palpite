""" Palpite web-app. """

import os

import pandas as pd
import streamlit as st

import helper
import keys
import palpite
import palpite.data
import palpite.draft

APP_NAME = "Palpite Cartola FC"
THIS_FOLDER = os.path.dirname(__file__)

# Page title and configs.
st.set_page_config(page_title=APP_NAME)
st.title(APP_NAME)

# Main inputs.
money = st.number_input("Cartoletas", min_value=0.0, value=100.0)

# TODO Add my team and exclude rivals option.

# Get clubs.
clubs = palpite.data.get_clubs_with_odds(
    key=keys.THE_ODDS_API, cache_folder=os.path.join(THIS_FOLDER, "cache")
)

# Initialize Cartola FC API.
cartola_fc_api = palpite.data.CartolaFCAPI()

# Players.
players = palpite.create_all_players(cartola_fc_api.players(), clubs)
# Keep only players that may play.
players = [player for player in players if player.status in [2, 7]]
# Keep only players from teams that have odds available.
players = [player for player in players if pd.notna(player.club.win_odds)]

# Schemes.
schemes = palpite.create_schemes(cartola_fc_api.schemes())

line_up = palpite.draft.draft(
    individuals=100,
    generations=100,
    players=players,
    schemes=schemes,
    max_price=money,
    tournament_size=5,
)

line_up_table = line_up.dataframe
line_up_table["Photo"] = line_up_table["Photo"].apply(helper.create_html_tag, height=60)
line_up_table = line_up_table[["Photo", "Predicted Points"]]

html_table = line_up_table.to_html(escape=False)
html_table = helper.format_html_table(html_table)
st.write(html_table, unsafe_allow_html=True)
