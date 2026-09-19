# mmhbmc-checkers

A fruit fly's mushroom body, wired from the FlyWire connectome, learning to
play checkers through dopamine-gated plasticity. Legal moves come from plain
code; the fly scores each candidate position and the best one is played.

## Layout

```
packages/flycore/   checkers rules, shared by both containers, stdlib only
services/brain/     fly brain and move choice: connectome, simulator, training
services/web/       website: board, moves, live brain activity
data/               FlyWire CSVs and connectome.npz, not in git
artifacts/          checkpoints and exported neuron metadata
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cd services/brain
../../.venv/bin/python -m connectome.build    # data/connectome.npz
../../.venv/bin/python -m connectome.export   # artifacts/neurons.json
../../.venv/bin/python -m sim.sparsity        # phase 2 gate
../../.venv/bin/python -m game.arena --games 200 --opponent random
../../.venv/bin/python -m train.selfplay --run first --games 20000
cd ../..
.venv/bin/pytest
```

`pytest -m "not slow"` skips deep perft and the full-connectome tests (~2 s
instead of ~55 s).

Details on the data, the mushroom body and the simulator are in
`services/brain/README.md`.

Long runs keep a one-line status per job in `artifacts/progress/` (training
runs are named after `--run`). To follow every job at once:

```bash
while true; do clear; cat artifacts/progress/*.txt; sleep 5; done
```

## Results: version 1

The plain fly (`artifacts/flies/plain.npz`, git tag `v1`) judges the board each
legal move leaves and plays the best one; it doesn't look further ahead.
Trained 4,096 games against a random player, then 2,048 against a greedy one,
it wins 69% of games against random and 73% against greedy (300 games each).
Minimax at depth 2 still beats it almost every time.

Does the fly's own wiring matter? Ten seeds per condition, 1,024 training games
each; gain is the improvement in win rate against random over each circuit's
untrained start (Welch's t-test against the real wiring):

| Mushroom body | Gain | p |
| --- | --- | --- |
| Real connectome | +9.7% ± 6.1 | |
| Shuffled, every neuron's degrees kept | +7.2% ± 5.6 | 0.36 |
| Random, same connection and synapse counts | +10.4% ± 7.2 | 0.82 |
| Real, but MBONs get another MBON's dopamine neurons | +1.0% ± 8.1 | 0.015 |

The specific wiring inside the mushroom body made no measurable difference;
pairing reward dopamine with the avoid MBONs did. That pairing is also how
approach and avoid are assigned here, so part of the effect is by
construction. With nine comparisons, p = 0.015 is suggestive rather than
decisive, and every condition shares our designed input stage.

Win/loss reward alone, without material won or lost after each move, made the
fly worse (44% against random): credit reaches only the last few moves. A large
win/loss dose credited to the whole game through a slow trace was worse still
(73% to 32% against random): nearly every Kenyon cell fires at some point in a
game, so the dose nudged almost every synapse the same way.

## Conventions

- Board positions are stored from the side to move, which always advances
  toward row 0. American rules: captures are mandatory, promotion ends a
  jump chain, kings don't fly.

## Compromises

- The board drives Kenyon cells directly. 128 input lines don't fit through
  the ~50 projection neurons of the real olfactory pathway.
- Starting weights are synapse counts, not measured strengths. Board input
  synapses are scaled up (2.4x) to reach the Kenyon cell sparsity of a real fly.
- Each Kenyon cell reads three squares on one diagonal, not random inputs, so
  it can detect local shapes.
- Board lines fire regular spike trains, not noisy ones.
- MBONs get a constant background input in place of the inputs from outside
  the mushroom body that aren't modelled.
- Signs come from predicted transmitters.
- No gap junctions, and only one hemisphere.
- Only KC→MBON synapses learn. Everything upstream keeps its connectome
  weights. Dopamine signals surprise and can both weaken and strengthen a
  synapse; the planned weakening-only rule stopped the fly learning.
