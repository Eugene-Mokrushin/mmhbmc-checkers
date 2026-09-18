# Training

Only KC→MBON synapses learn (`plasticity.py`), and only from Kenyon cells that
were recently active. Dopamine carries surprise: the outcome minus what the
fly's own score predicted. Better than expected weakens those synapses in
reward (PAM) compartments, where the avoid MBONs are, and strengthens them in
punishment (PPL1) compartments; worse than expected does the opposite (Hige et
al. 2015, Handler et al. 2019). Weights stay within [0, 2x] of their start and
drift slowly back.

Two earlier rules failed in instructive ways. Strengthening every silent cell
alongside the active ones blurred the few cells that detect a shape, capping
"open to capture" at 0.59 AUC. Only weakening active cells drained the MBONs'
input until they fired the same for every board, so the score stopped
reflecting anything learned.

## Sanity check

`python -m train.sanity` shows positions from random games and delivers
dopamine by material balance, then measures on held-out positions how well the
fly's score separates "ahead" from "behind" (AUC, 0.5 = chance). Over four
passes it goes from 0.51 to 0.84. Trained the same way on whether a move leaves
a piece open to capture, it goes from 0.54 to 0.72.

## Games

```bash
python -m train.selfplay --run NAME --games 20000 --opponent random --reward shaped
```

Plays many games at once. Each fly move adds the chosen board's active Kenyon
cells to an eligibility trace that keeps 60% per move. After the opponent
replies, the reward is the material won or lost (`shaped`), or nothing until
the end (`terminal`). A win is +3, a loss -3.

Every `--eval-every` games it saves the weights and plays the frozen fly
against random, greedy, minimax (depth 2) and its own untrained self. Results
go to `artifacts/runs/NAME/log.csv`, checkpoints to
`artifacts/runs/NAME/fly-*.npz`, and the live status to `artifacts/progress/NAME.txt`.

Those in-run evaluations use 100 games, so each point is only good to about
±10 points. `python -m train.curve --run NAME --games 300` replays every
checkpoint afterwards for a cleaner learning curve.

## Controls

Phase 5 asks whether the fly's wiring matters or any similar circuit would do.
`--control` trains the same pipeline on a different circuit
(`connectome/controls.py`):

- `degree`: connections shuffled so every neuron keeps its in- and out-degree
  and every connection its weight.
- `random`: the same number of connections and synapses, placed at random.
- `compartments`: real wiring, but each MBON gets another MBON's dopamine
  neurons, so reward no longer lands on the avoid MBONs.

Shuffles stay inside each population pair (KC→MBON, APL→KC, ...). Mixed
freely, APL would inhibit random cells and the control would fail for a
trivial reason. Both rewired circuits pass the same sparsity gate as the real
one (6-9% of Kenyon cells active).

Ten seeds per condition, then the comparison:

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for control in real degree random compartments; do
    python -m train.selfplay --run c-$control-$seed --control $control --seed $seed \
      --games 2048 --eval-every 2048
  done
done
python -m train.report --pattern "c-*"
```

The report gives each condition's final win rates and its gain over its own
untrained start, with Welch's t-test against the real wiring.

## Skills and freezing

Every evaluation also scores the fly on a fixed exam of 1,500 boards
(`skills.py`): how well its score separates safe boards from ones where the
opponent can capture (`safety`), and ahead from behind in material
(`material`), both as AUC. That is far less noisy than 100-game win rates.

`python -m train.freeze --runs RUN [RUN ...] --name plain` re-tests the best
three checkpoints over 300 games and saves the winner to
`artifacts/flies/plain.npz` with its settings and git commit; `load_fly("plain")`
rebuilds it exactly. `--init RUN` or `--init RUN@GAMES` continues training from
a saved fly.

## Finishing version 1 overnight

```bash
train/night.sh BEST_RUN RATE CONTROL_GAMES
```

Trains the best fly on against greedy and, alongside, a fresh fly with
win/loss-only reward; freezes `plain` and tags `v1`; then runs the controls,
two at a time, and writes `artifacts/runs/controls-report.txt`. It keeps the
Mac awake while it runs (it still sleeps if the lid is closed). Timestamps go
to `artifacts/runs/night.log`.
