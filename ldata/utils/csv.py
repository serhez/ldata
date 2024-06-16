import csv
from typing import Any, Type

import numpy as np
import numpy.typing as npt

CSV_OPTS = {
    "delimiter": ",",
    "quotechar": '"',
    "escapechar": "\\",
}

_column_error_msg = (
    "Error while reading CSV file: requested column not found in headers."
)


def read_csv_columns(
    file: Any, columns: list[str], dtype: Type[Any] = str, error_msg: str | None = None
) -> tuple[npt.NDArray[Any], ...]:
    """
    Read the requested columns from a CSV file.

    ### Parameters
    ----------
    `file`: the file object to read from.
    `columns`: a list of strings with the names of the columns to read.
    `dtype`: the type of the data to read.
    `error_msg`: a message to raise in case of a "column not found" error.
    - If `error_msg` is `None`, a default message will be raised.

    ### Returns
    ----------
    A tuple with the data from the requested columns, in the same order as requested.

    ### Raises
    ----------
    `ValueError`: if the requested columns are not found in the headers.
    """

    rows = list(csv.reader(file, **CSV_OPTS))  # type: ignore[reportArgumentType]

    # Find columns indeces
    headers = rows[0]
    try:
        columns_idxs = [headers.index(column) for column in columns]
    except ValueError:
        if error_msg is not None:
            raise ValueError(error_msg)
        raise ValueError(_column_error_msg)

    # Return the data from the requested columns
    return tuple(
        np.array([dtype(row[i].strip()) for row in rows[1:]]) for i in columns_idxs
    )


def write_csv(file: Any, headers: list[str], data: npt.NDArray[Any]):
    """
    Write data to a CSV file.

    ### Parameters
    ----------
    `file`: the file object to write to.
    `headers`: the headers of the CSV file.
    `data`: the data to write.
    """

    writer = csv.writer(file, **CSV_OPTS)  # type: ignore[reportArgumentType]
    writer.writerow(headers)
    writer.writerows(data)
