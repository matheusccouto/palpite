""" Web app helper functions. """

import palpite


def format_html_table(html):
    """ Format a html table. """
    html = html.replace("<table", "<table width=100%")
    html = html.replace("<table", '<table style="text-align: center;"')
    return html


def create_html_tag(photo: str, height: int):
    """ Create html tag with image. """
    return f'<img src="{photo}" height="{height}">'

