# Brain service

The fly brain and move choice: connectome, simulator, training.

## Data

FlyWire FAFB v783 from codex.flywire.ai, downloaded 2026-09-18, into
`data/raw/`:

```
classification.csv.gz                        139,255 rows
consolidated_cell_types.csv.gz               138,327
neurons.csv.gz                               139,255   transmitter predictions
connections_princeton_no_threshold.csv.gz 22,285,323
connections_princeton.csv.gz               5,342,446   >=5 synapses, tests only
coordinates.csv.gz                           238,909
```

Codex regenerated some of these in July 2025 without changing the version
label, so the SHA-256 of every file is pinned in
`services/brain/connectome/raw.py` and the build refuses anything else.

The model uses the unthresholded graph: 19,773,733 connected pairs and
76,944,499 synapses. At FlyWire's usual 5-synapse cutoff it is 3,732,460
pairs and 50,666,648 synapses, identical to Codex's filtered file.

Things that are easy to get wrong:

- There is one connection row per pair per neuropil. Sum the pair before
  applying any threshold.
- The connections table gives every neuron with outputs exactly one
  transmitter. It agrees with `neurons.csv` wherever both have one, and it
  covers 19,406 neurons that `neurons.csv` leaves blank. 252 neurons have no
  outputs at all.
- There are no verified transmitter labels in these files. DPM is predicted
  dopaminergic, but the literature has it as serotonergic/GABAergic.
- Coordinates are proofreading markers, up to 177 per neuron, not somas.
- FAFB images are mirrored left/right. Side labels are from the fly's point
  of view and are correct.

## Mushroom body

The model uses the right mushroom body with no synapse cutoff: 2,597 Kenyon
cells, 48 MBONs, 309 DANs and 1 APL, joined by 490,752 connections.
`artifacts/neurons.json` lists them in model order with one position each,
for the website.

Side labels mark where a cell body sits, and 16 MBONs have theirs across the
midline from the lobe they read. So membership comes from Kenyon cells, which
never cross: an MBON or APL belongs to the mushroom body holding most of its
KC synapses. Most DANs split their KC synapses between both mushroom bodies,
so a DAN is included if at least 10% are here. Compartments come from which
DANs contact which MBONs, not from a hand-written table. DPM is left out.

## Simulator

Current-based leaky integrate-and-fire neurons with Shiu et al. 2024 constants
(`sim/params.py`), stepped at 0.1 ms. `sim/reference.py` is the Brian2 model,
`sim/fast.py` the PyTorch one used everywhere else. On the right mushroom body
the two produce identical spike rasters (`tests/test_equivalence.py`).

The fast simulator only sends weights from neurons that spiked on a step, so
one position takes about 80 ms, 8 candidates about 0.3 s, and batches of 64
about 14 ms per position on a laptop CPU.

Each Kenyon cell keeps its real projection-neuron claws (median 6), with their
synapse counts, rewired to random board lines. 126 Kenyon cells have no such
input and are never driven by the board.

`python -m sim.sparsity` runs the sparsity gate. PN->KC synapses are 1.75x the
standard weight, which puts 5-10% of Kenyon cells in play for 12-24 active
lines, with different inputs sharing few of them. Without APL the whole layer
fires from 8 lines up.

Known weak spots:

- Endgames with 8 or fewer pieces light only ~3% of Kenyon cells.
- The same board run twice shares about half its active Kenyon cells.
- MBONs barely fire (a few Hz). A third of their real input comes from
  outside the mushroom body and isn't modelled.

## Conventions

- Neurons are indexed by ascending root ID. `counts[pre, post]` holds
  synapse counts; `weights()` multiplies each row by the presynaptic sign.
- Sign follows Shiu et al. 2024: ACh, DA, 5-HT and octopamine excite;
  GABA and glutamate inhibit.
