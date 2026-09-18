import pytest

from flycore.board import INITIAL
from flycore.moves import perft

PUBLISHED = [7, 49, 302, 1469, 7361, 36768, 179740, 845931, 3963680]


@pytest.mark.parametrize("depth", range(1, 7))
def test_perft(depth):
    assert perft(INITIAL, depth) == PUBLISHED[depth - 1]


@pytest.mark.slow
@pytest.mark.parametrize("depth", [7, 8, 9])
def test_perft_deep(depth):
    assert perft(INITIAL, depth) == PUBLISHED[depth - 1]
