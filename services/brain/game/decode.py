import numpy as np

from connectome.extract import MushroomBody


def valence(mb: MushroomBody) -> np.ndarray:
    # Aso et al. 2014: MBONs in reward (PAM) compartments drive avoidance,
    # MBONs in punishment (PPL1) compartments drive approach.
    pam = np.strings.startswith(mb.graph.cell_type[mb.members("DAN")], "PAM")
    reward_share = (mb.dan_mbon_contact() * pam[:, None]).sum(axis=0)
    return np.where(reward_share > 0.5, -1, 1)


def score(counts: np.ndarray, mb: MushroomBody, signs: np.ndarray) -> np.ndarray:
    return counts[:, mb.members("MBON")] @ signs
