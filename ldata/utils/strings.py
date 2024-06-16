# Dictionary mapping whitespace characters to their escaped representations
_whitespace_escape_map = {
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
    "\f": "\\f",
    "\v": "\\v",
}


def escape_whitespace(input_string: str) -> str:
    """
    Replace whitespace characters in a string with their escaped versions.

    ### Parameters
    --------------
    `input_string`: the string to escape.

    ### Returns
    --------------
    A new string with all whitespace characters replaced with their escaped versions.
    """

    # Initialize an empty list to hold the escaped characters
    escaped_string = []

    # Iterate through each character in the input string
    for char in input_string:
        # If the character is a whitespace character, replace it with its escaped version
        if char in _whitespace_escape_map:
            escaped_string.append(_whitespace_escape_map[char])
        else:
            escaped_string.append(char)

    # Join the list into a single string and return
    return "".join(escaped_string)


def enclose_in_quotes(input_string: str) -> str:
    """
    Enclose a string in double quotes.

    ### Parameters
    --------------
    `input_string`: the string to enclose in quotes.

    ### Returns
    --------------
    A new string with the input string enclosed in double quotes.
    """

    # Replace all double quotes with escaped double quotes
    input_string = input_string.replace('"', '\\"')
    input_string = input_string.replace('\\\\"', '\\"')
    return f'"{input_string}"'


def stripped_string_compare(s1: str, s2: str) -> bool:
    """
    Compare two strings after stripping leading and trailing whitespace.

    ### Parameters
    --------------
    `s1`: the first string to compare.
    `s2`: the second string to compare.

    ### Returns
    --------------
    `True` if the stripped strings are equal, `False` otherwise.
    """

    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()

    return s1 == s2
