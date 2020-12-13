""" Palpiteiro web-app. """

import os
import time

import pandas as pd
import streamlit as st

import helper
import palpiteiro.data
import palpiteiro.draft

# Constants.
APP_NAME = "Palpiteiro"
SUBTITLE = "Recomendação de escalações para o Cartola FC"
THIS_FOLDER = os.path.dirname(__file__)
FAVICON = os.path.join("img", "soccerball.png")

# Page title and configs.
st.set_page_config(page_title=APP_NAME, page_icon=FAVICON)
st.title(APP_NAME)
st.text(SUBTITLE)

# Main inputs.
st.sidebar.title("Configurações")
money = st.sidebar.number_input("Cartoletas", min_value=0.0, value=100.0, format="%.1f")

# Get clubs.
key = os.environ.get("THE_ODDS_API")
clubs = palpiteiro.data.get_clubs_with_odds(
    key=key, cache_folder=os.path.join(THIS_FOLDER, "cache")
)

# Initialize Cartola FC API.
cartola_fc_api = palpiteiro.data.CartolaFCAPI()

# Players.
players = palpiteiro.create_all_players(cartola_fc_api.players(), clubs)
# Keep only players that may play.
players = [player for player in players if player.status in [2, 7]]
# Keep only players from teams that have odds available.
players = [player for player in players if pd.notna(player.club.win_odds)]

# Schemes.
schemes = palpiteiro.create_schemes(cartola_fc_api.schemes())

# Select teams.
clubs_names = sorted(clubs.dropna(subset=["win_odds"])["nome"])
selected_clubs = st.sidebar.multiselect(
    "Times", options=clubs_names, default=clubs_names
)
players = [player for player in players if player.club.name in selected_clubs]

# Select schemes.
schemes = st.sidebar.multiselect("Esquemas Táticos", options=schemes, default=schemes)

# About the app
with open(os.path.join(THIS_FOLDER, "SOBRE.md"), encoding="utf-8") as file:
    about = file.read()
st.sidebar.markdown(about)

# Exceptions handling.

# If there isn't no games.
if len(clubs_names) == 0:
    st.error(
        "Não foi possível preparar uma escalação porque "
        "ainda não há cotações disponíveis para as próximas partidas. "
        "Tente novamente mais tarde."
    )
    st.stop()

# If the player selected no clubs.
if len(selected_clubs) == 0:
    st.error("Você deve selecionar pelo menos um time.")
    st.stop()

# If the player selected no schemes.
if len(schemes) == 0:
    st.error("Você deve selecionar pelo menos uma formação tática.")
    st.stop()

# Get line up.
if st.button("Escalar"):
    with st.spinner("Por favor aguarde enquanto o algoritmo escolhe os jogadores..."):
        try:
            line_up = palpiteiro.draft.draft(
                individuals=100,
                generations=100,
                players=players,
                schemes=schemes,
                max_price=money,
                tournament_size=5,
            )
        except RecursionError:
            st.error(
                "Não foi possível montar um escalação para esta quantidade de cartoletas "
                "com os times e formações táticas selecionados. "
                "Experimente adicionar mais time e formações táticas, "
                "ou aumentar a quantidade de cartoletas."
            )
            st.stop()

    # Show line up.
    st.header("Aqui está a sua escalação")

    # Arrange data.
    line_up_table = line_up.dataframe
    line_up_table["Player"] = line_up_table.apply(
        lambda x: helper.create_html_tag(photo=x["Photo"], name=x["Name"], height=32),
        axis=1,
    )
    line_up_table["Club"] = line_up_table.apply(
        lambda x: helper.create_html_tag(photo=x["Club Photo"], height=32), axis=1,
    )
    line_up_table = line_up_table[["Position Name", "Club", "Player"]]

    # Transform into html and show on app.
    html_table = line_up_table.to_html(
        escape=False, header=False, index=False, border=0
    )
    html_table = helper.format_html_table(html_table)
    st.write(html_table, unsafe_allow_html=True)

    # General info.
    st.title("")
    st.header("Informações Gerais")
    st.text(f"Esquema Tático\t\t{line_up.scheme}")
    st.text(f"Pontuação Esperada\t{line_up.predicted_points:.1f} pontos")
    st.text(f"Custo Total\t\t{line_up.price:.1f} cartoletas")

    st.header("Partidas Consideradas")
    st.text(
        palpiteiro.data.get_matches(
            key=key,
            strf="%d/%m/%y",
            clubs=clubs,
            cache_folder=os.path.join(THIS_FOLDER, "cache"),
        )
    )
