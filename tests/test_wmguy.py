import pytest
from src.core.wmguy import WMGuy

def test_wmguy_initialization():
    wmguy = WMGuy()
    assert wmguy.config is not None
    assert wmguy.memory is not None
    assert wmguy.learning is not None 