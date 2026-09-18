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
    # PN->KC synapses relative to w_syn, tuned so every stage of a game activates ~7% of KCs
    input_gain: float = 2.4
    # background input to MBONs, about 10 Hz at rest
    mbon_drive: float = 7.3e-3

    @property
    def ref_steps(self) -> int:
        return round(self.t_ref / self.dt)

    @property
    def input_period(self) -> int:
        return round(1 / (self.input_rate * self.dt))
