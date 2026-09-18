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

- Each Kenyon cell keeps its real projection-neuron claws (median 6) and their
  synapse counts, rewired to random board lines. 126 Kenyon cells have none.
- Every piece lights one line, which fires a regular 150 Hz train with its own
  fixed offset. The same board always gives the same spikes, so the fly's
  score for a position is repeatable.
- Input synapses are 2.5x the standard weight, scaled by sqrt(24 / pieces) the
  way the antennal lobe normalizes odor strength. That keeps 7-9% of Kenyon
  cells active from opening to endgame (`python -m sim.sparsity`). Without APL
  the whole layer fires.
- MBONs get a constant 7.3 mV background input standing in for the third of
  their synapses that come from outside the mushroom body, which gives them a
  resting rate around 12 Hz. Every evaluation starts from a settled resting
  state rather than from silence.
