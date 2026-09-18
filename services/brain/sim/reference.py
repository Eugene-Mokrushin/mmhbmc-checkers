import brian2 as b2
import numpy as np
import scipy.sparse as sp

from sim.params import LIF

EQUATIONS = """
dv/dt = (x + bias - (v - v_rest)) / tau_m : volt (unless refractory)
dx/dt = -x / tau_s : volt
bias : volt (constant)
"""


def raster(weights: sp.csr_array, projection: sp.csr_array, inputs: np.ndarray, p: LIF = LIF(), bias=None) -> np.ndarray:
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = p.dt * b2.second
    steps, n_lines = inputs.shape
    n = weights.shape[0]
    namespace = {
        "v_rest": p.v_rest * b2.volt,
        "v_reset": p.v_reset * b2.volt,
        "v_th": p.v_th * b2.volt,
        "tau_m": p.tau_m * b2.second,
        "tau_s": p.tau_s * b2.second,
    }
    neurons = b2.NeuronGroup(
        n,
        EQUATIONS,
        threshold="v > v_th",
        reset="v = v_reset",
        refractory=p.t_ref * b2.second,
        method="exact",
        namespace=namespace,
    )
    neurons.v = p.v_rest * b2.volt
    if bias is not None:
        neurons.bias = np.asarray(bias) * b2.volt

    step, line = np.nonzero(inputs)
    board = b2.SpikeGeneratorGroup(n_lines, line, step * p.dt * b2.second)
    synapses = []
    for source, m, gain in [(neurons, weights, 1.0), (board, projection, p.input_gain)]:
        m = sp.coo_array(m)
        s = b2.Synapses(source, neurons, "w : volt", on_pre="x_post += w")
        s.connect(i=m.row, j=m.col)
        s.w = m.data * p.w_syn * gain * b2.volt
        synapses.append(s)

    monitor = b2.SpikeMonitor(neurons)
    b2.Network(neurons, board, *synapses, monitor).run(steps * p.dt * b2.second)
    out = np.zeros((steps, n), dtype=bool)
    out[np.round(monitor.t / (p.dt * b2.second)).astype(int), monitor.i[:]] = True
    return out
