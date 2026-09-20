import numpy as np

from connectome.extract import MushroomBody


def reward_share(mb: MushroomBody) -> np.ndarray:
    # per MBON, the share of its DAN contact that comes from reward (PAM) DANs
    # rather than punishment (PPL1) DANs
    pam = np.strings.startswith(mb.graph.cell_type[mb.members("DAN")], "PAM")
    return (mb.dan_mbon_contact() * pam[:, None]).sum(axis=0)


def valence(mb: MushroomBody) -> np.ndarray:
    # Aso et al. 2014: MBONs in reward compartments drive avoidance,
    # MBONs in punishment compartments drive approach.
    return np.where(reward_share(mb) > 0.5, -1, 1)

