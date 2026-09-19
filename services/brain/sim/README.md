# Simulator

Current-based leaky integrate-and-fire neurons with Shiu et al. 2024 constants
(`sim/params.py`), stepped at 0.1 ms. `sim/reference.py` is the Brian2 model,
`sim/fast.py` the PyTorch one used everywhere else. On the right mushroom body
the two produce identical spike rasters (`tests/test_equivalence.py`).

The fast one looks at a board for 50 ms (500 steps). Only neurons that spiked
send anything, and only along their real connections; the per-step update is
compiled into one pass (`sim/kernels.py`). In batches of 1,024 a position costs
about 3 ms on an M2 laptop. What's left is mostly fixed overhead per step.

Board input, in `sim/inputs.py`:

- Every square lights one of five lines: own man, own king, their man, their
  king, or empty (160 lines, `flycore/encode.py`). Each line fires a regular
  150 Hz train with its own fixed offset, so the same board always gives the
  same spikes and the fly's score for a position is repeatable.
- Each Kenyon cell keeps its real projection-neuron claws (median 6) and their
  synapse counts, but they read the 15 lines of three squares on one diagonal.
  A cell can then fire for a local shape such as "my man, their man, empty
  square behind", which is what spotting a jump needs. With claws scattered
  over the whole board, the best possible readout of that shape was 0.75 AUC;
  with local fields it is 0.95. 126 Kenyon cells have no claws; two with more
  than 15 keep their strongest 15.
- Input synapses are 2.4x the standard weight, which keeps about 7% of Kenyon
  cells active at every stage of a game (`python -m sim.sparsity`). Different
  positions share more active cells than they did with scattered claws
  (overlap 0.3-0.45), because common local shapes recur. Without APL the whole
  layer fires.
- MBONs get a constant 7.3 mV background input standing in for the third of
  their synapses that come from outside the mushroom body, which gives them a
  resting rate around 12 Hz. Every evaluation starts from a settled resting
  state rather than from silence.

## Whole brain

`FastLIF` keeps weights only as sparse rows and takes a `device`, so the same
code runs all 139,255 neurons, on the CPU or a CUDA GPU (`best_device()`;
Apple's GPU gives identical spikes but no speed-up). The whole brain uses
connections of 5 or more synapses (3.7M).

`python -m sim.feeding` is its validation gate, after Shiu et al. 2024: the
sugar and bitter taste neurons are driven at 150 Hz and MN9, the motor neuron
behind feeding, is watched. Sugar alone drives it at about 400 Hz from
silence, bitter alone leaves it silent, and bitter together with sugar cuts it
by about 80%. With every connection down to a single synapse, sugar lights up
12% of the brain and bitter no longer holds MN9 back, so the weakest
connections stay out.

## Eyes

`sim/eye.py` shows the fly the board. Each playable square has a brightness for
its state (empty, own man or king, opponent man or king) and the board fills
the frontal field, each eye seeing its half. Every photoreceptor (R1-6, R7, R8;
11,151 of them) fires steadily at a rate set by the brightness where it looks.
Where it looks comes from the lamina's geometry for R1-6 (the retina isn't in
the data) and, for R7 and R8, from the lamina and medulla neurons they share
with them. Elevation is reliable; azimuth is only approximate until FlyWire's
published column coordinates are used.

`python -m sim.seeing` measures how much of the board each stage of the brain
still carries: the best linear readout of material and safety from its spikes.
Material survives from the eyes (0.99 AUC) to the Kenyon cells, MBONs and
descending neurons (about 0.8). Safety, a pattern rather than a sum, is 0.67 at
the eyes and gone by the Kenyon cells: only 176 of 2,597 right-side Kenyon
cells receive visual input at all.
