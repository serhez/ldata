import numpy as np
import requests
from bs4 import BeautifulSoup


def get_common_names() -> list[str]:
    """
    Get a list of 1,690 single-word most common names from forebears.io.
    Requires internet connection.

    ### Returns
    ----------
    A list of the names.

    ### Raises
    ----------
    `ConnectionError`: if the request fails.

    ### Notes
    ----------
    - The names contain only ascii characters.
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
        if all(c not in name for c in sep_chars)
        and name.isalpha()
        and name.isascii()
        and len(name) >= 4
    ]

    return names


def get_random_strings() -> list[str]:
    """
    Get a list of 2,000 auto-generated random strings.

    ### Returns
    ----------
    A list of the strings.

    ### Notes
    ----------
    - The strings contain only ascii characters.
    - The strings are at least 4 characters long and at most 8 characters long.
    """

    np.random.seed(0)
    strings = [
        "".join(
            np.random.choice(
                list("abcdefghijklmnopqrstuvwxyz"), np.random.randint(4, 9)
            )
        )
        for _ in range(2000)
    ]

    return strings


def get_random_words() -> list[str]:
    """
    Get a list of 10,000 most common English words from mit.edu.
    Requires internet connection.

    ### Returns
    ----------
    A list of the words.

    ### Raises
    ----------
    `ConnectionError`: if the request fails.

    ### Notes
    ----------
    - The words contain only ascii characters.
    - The words are at least 4 characters long.
    """

    response = requests.get("https://www.mit.edu/~ecprice/wordlist.10000")
    words = response.content.splitlines()

    # Remove spaces from the words
    words = [word.decode("utf-8") for word in words]
    words = [
        word
        for word in words
        if len(word) >= 4 and all(char.isalpha() for char in word)
    ]

    return words
