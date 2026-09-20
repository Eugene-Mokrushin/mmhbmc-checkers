# Playing

`game/fly.py` scores the board each legal move leaves behind: approach MBON
spikes minus avoid MBON spikes over 50 ms, ties broken at random. Valence comes
from compartments (Aso et al. 2014): MBONs mostly contacted by reward (PAM)
DANs drive avoidance, those contacted by punishment (PPL1) DANs drive approach.
That gives 23 approach and 25 avoid MBONs.

`python -m game.arena --games N --opponent random|greedy|minimax` plays the fly
against a baseline, many games at once. Because captures are mandatory, the
one-move greedy player is barely better than random; minimax at depth 2 is the
first real step up. Games are drawn after 200 plies.

## The whole fly

`game/wholefly.py` is version 2: all 139,255 neurons, the board seen through the
eyes for 100 ms (it takes that long to reach the mushroom bodies) and scored the
same way, from the MBONs of both mushroom bodies (96, 5,177 Kenyon cells). The
brain keeps connections of 5 or more synapses, except between Kenyon cells and
MBONs, where every synapse stays because that is where it learns. It judges a
board in about 0.4 s on the laptop. `python -m train.selfplay --brain whole`
trains it with the same dopamine rule; the rewired controls exist only for the
mushroom-body fly.

## The trained whole fly

`game/gradfly.py` is version 2's other variant: the same brain and eyes, but
every connection's strength is learned by gradient descent rather than by the
fly's own dopamine; signs and wiring stay FlyWire's. A board's score is a learned
weighted sum of descending-neuron spikes over 100 ms. It is a neural network
shaped like a fly brain, not a model of how a fly learns.

It learns in two stages. `python -m train.teacher` has players of mixed
strength play each other and a depth-7 search score every legal move in about
110,000 of their positions. `python -m train.gradient` teaches the brain to rank
moves as the search does and to give each board a value that means the same
everywhere, so it can also judge lines it imagines. `python -m train.outcomes`
then has it play minimax and greedy players and learn from how its own games
ended, with teacher lessons mixed in so it keeps them. At play time there is no
search: the fly decides alone.
