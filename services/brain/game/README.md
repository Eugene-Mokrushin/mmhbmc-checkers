# Playing

`game/fly.py` scores the board each legal move leaves behind: approach MBON
spikes minus avoid MBON spikes over 100 ms, ties broken at random. Valence comes
from compartments (Aso et al. 2014): MBONs mostly contacted by reward (PAM)
DANs drive avoidance, those contacted by punishment (PPL1) DANs drive approach.
That gives 23 approach and 25 avoid MBONs.

`python -m game.arena --games N --opponent random|greedy|minimax` plays the fly
against a baseline, many games at once. Because captures are mandatory, the
one-move greedy player is barely better than random; minimax at depth 2 is the
first real step up. Games are drawn after 200 plies.
