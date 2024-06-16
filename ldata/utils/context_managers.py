import sys
from io import StringIO


class Capturing(list):
    """
    Used to capture stdout as a list.
    From https://stackoverflow.com/a/16571630/6416660.
    Alternative use redirect_stdout() from contextlib.
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1  # type: ignore[reportAttributeAccessIssue]
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
