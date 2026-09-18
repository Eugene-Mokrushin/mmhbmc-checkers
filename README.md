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
cd ../..
.venv/bin/pytest
```

`pytest -m "not slow"` skips deep perft and the full-connectome tests (~2 s
instead of ~55 s).

Details on the data, the mushroom body and the simulator are in
`services/brain/README.md`.

Long runs rewrite a one-line status to `artifacts/progress.txt`:

```bash
watch cat artifacts/progress.txt
```

## Conventions

- Board positions are stored from the side to move, which always advances
  toward row 0. American rules: captures are mandatory, promotion ends a
  jump chain, kings don't fly.

## Compromises

- The board drives Kenyon cells directly. 128 input lines don't fit through
  the ~50 projection neurons of the real olfactory pathway.
- Starting weights are synapse counts, not measured strengths. Board input
  synapses are scaled 1.75x to reach the Kenyon cell sparsity of a real fly.
- Signs come from predicted transmitters.
- No gap junctions, and only one hemisphere.
- Only KC→MBON synapses learn. Everything upstream keeps its connectome
  weights.
