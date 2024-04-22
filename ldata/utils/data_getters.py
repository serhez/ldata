import requests
from bs4 import BeautifulSoup


def get_common_names() -> list[str]:
    """
    Get a list of 1,708 single-word most common names from forebears.io.
    Requires internet connection.

    ### Returns
    ----------
    A list of the names.

    ### Raises
    ----------
    `ConnectionError`: if the request fails.

    ### Notes
    ----------
    - The names do not contain non-alphabetic characters.
    - The names are at least 4 characters long.
    """

    response = requests.get(
        "https://forebears.io/earth/forenames", headers={"User-Agent": "Custom"}
    )
    soup = BeautifulSoup(response.text, "html.parser")
    names = [a.text for a in soup.select("a[href^='forenames/']")]

    response = requests.get(
        "https://forebears.io/earth/surnames", headers={"User-Agent": "Custom"}
    )
    soup = BeautifulSoup(response.text, "html.parser")
    names.extend([a.text for a in soup.select("a[href^='surnames/']")])

    # Remove unwanted words
    sep_chars = [" ", "-", "'"]
    names = [
        name
        for name in names
        if all(c not in name for c in sep_chars) and name.isalpha() and len(name) >= 4
    ]

    return names
