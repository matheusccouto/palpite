""" Web app helper functions. """

import re
from typing import Optional


def format_html_table(html):
    """ Format a html table. """
    html = html.replace("<table", "<table width=100%")
    html = html.replace("<table", '<table style="text-align: left; border-collapse: collapse"')
    html = html.replace('<td', '<td style="border: none"')
    html = html.replace('<tr', '<tr style="border: none"', 1)
    return html


def create_html_tag(photo: str, height: int, name: Optional[str] = None):
    """ Create html tag with image. """
    if name:
        name = f" {name}"
    else:
        name = ""
    return f'<img src="{photo}" height="{height}">{name}'
