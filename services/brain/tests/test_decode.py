from connectome.extract import extract
from game.decode import valence


def test_mbon_in_a_reward_compartment_signals_avoid(brain):
    assert valence(extract(brain)).tolist() == [-1]

