from dataclasses import dataclass


@dataclass(frozen=True)
class LIF:
    # Shiu et al. 2024, SI units
    dt: float = 1e-4
    v_rest: float = -52e-3
    v_reset: float = -52e-3
    v_th: float = -45e-3
    tau_m: float = 20e-3
    tau_s: float = 5e-3
    t_ref: float = 2.2e-3
    w_syn: float = 0.275e-3
    input_rate: float = 150.0
    # PN->KC synapses relative to w_syn, tuned so 9-24 piece boards activate 5-10% of KCs
    input_gain: float = 1.75

    @property
    def ref_steps(self) -> int:
        return round(self.t_ref / self.dt)
