"""Unclipped floating input, binary hidden p-bits, continuous final readout."""
from full_path_pdnn_ffn import FullPathPBitFFN


class ContinuousInputPBitFFN(FullPathPBitFFN):
    def __init__(self, config):
        if (config.input_encoding != "continuous_raw" or config.coding != "binary"
                or config.architecture != "serial" or config.learnable_thresholds):
            raise ValueError("continuous_raw requires serial binary coding without separate thresholds")
        super().__init__(config)
        if hasattr(self, "input_temperature_raw"):
            del self.input_temperature_raw

    def input_mean(self, hidden_states):
        # Preserve the actual activation and its upstream gradient, including tails.
        return hidden_states

    def sampled_path_forward(self, hidden_states):
        state = hidden_states
        for index, projection in enumerate(self.projections[:-1]):
            state = self._sample_state(self.hidden_mean(projection(state), index))
        return self.projections[-1](state)
