# Training

Only KC→MBON synapses learn (`plasticity.py`). Dopamine weakens synapses from
Kenyon cells that were recently active (Hige et al. 2015) and strengthens those
from silent ones (Cohn et al. 2015), within [0, 2x] of the starting weight. A
reward reaches MBONs in PAM compartments and a punishment those in PPL1
compartments, through each MBON's real DAN contacts. So a reward turns down the
avoid MBONs for the boards that earned it, and a punishment turns down the
approach MBONs.

Depression alone, as first planned, ran synapses down until nothing was left
to learn with. On the sanity task it peaked at 0.75 and then decayed.

## Sanity check

`python -m train.sanity` shows positions from random games and delivers
dopamine by material balance, then measures on held-out positions how well the
fly's score separates "ahead" from "behind" (AUC, 0.5 = chance). It goes from
0.47 to 0.89 in three passes over 1,024 positions.

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
`artifacts/runs/NAME/fly-*.npz`, and the live status to `artifacts/progress.txt`.

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
