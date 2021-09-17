import numpy as np
from typing import Tuple


class WavePropagationLayer:

    def __init__(self, shape: Tuple[int], randomize_neurons: bool, randomize_synapses: float):
        self.shape = shape

        if randomize_neurons:
            # make results reproducible
            np.random.seed(7)
            # excitatory neurons
            re = np.random.uniform(0, 1, shape)
            # inhibitory neurons
            ri = np.random.uniform(0, 1, shape)
        else:
            re = np.zeros(shape)
            ri = np.ones(shape)

        # time scale of the recovery variable u
        self._a = np.asarray([0.02 * np.ones(shape), 0.02 + 0.08 * ri])
        # sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v.
        self._b = np.asarray([0.2 * np.ones(shape), 0.25 - 0.05 * ri])
        # after-spike reset value of the membrane potential v caused by fast high-threshold K+ conductances.
        self._c = np.asarray([-65 + 15 * re**2, -65 * np.ones(shape)])
        # after-spike reset of the recovery variable u caused by slow high-threshold Na+ and K+ conductances.
        self._d = np.asarray([8 - 6 * re**2, 2*np.ones(shape)])

        self._v = -65 * np.ones((2, *shape))
        self._u = self._b*self._v

        # construct synaptic connection matrix
        self._S = np.zeros((2, *shape) * 2)

        suppression_range = 1
        excitation_range = 2

        # excitation layer
        self._S = self._imprint_circular_kernel(self._S, layer_from=0, layer_to=0, radius=excitation_range, max_value=1, center_value=0)
        # inhibition layer
        self._S = self._imprint_circular_kernel(self._S, layer_from=0, layer_to=1, radius=suppression_range, max_value=0.5, center_value=0)
        # inhibition deactivates a local cluster
        self._S = self._imprint_circular_kernel(self._S, layer_from=1, layer_to=0, radius=suppression_range, max_value=-9, center_value=-9)
        # some rescaling as connections between nodes are sparse
        self._S *= 50

        if randomize_synapses > 0.:
            # make results reproducible
            np.random.seed(42)
            # vary synaptics strength randomly by +/- setup['randomize_synapses'] (relative)
            rs = np.random.uniform(1.-randomize_synapses, 1.+randomize_synapses, self._S.shape)
            self._S *= rs

    @staticmethod
    def _imprint_circular_kernel(field: np.array, layer_from: int, layer_to: int, radius: int, max_value: float,
                                 center_value: float, center_radius: int = 0, power: int = 1) -> np.array:

        assert field.shape[1:3] == field.shape[-2:]

        for row in range(field.shape[-2]):
            row_range = list(range(max(0, row-radius), min(field.shape[-2], row+radius+1)))

            for col in range(field.shape[-1]):
                col_range = list(range(max(0, col-radius), min(field.shape[-1], col+radius+1)))

                for d_row in row_range:
                    for d_col in col_range:
                        delta = np.sqrt((row-d_row)**2 + (col-d_col)**2)

                        if power == 0 and delta > radius:
                            field[layer_to, d_row, d_col, layer_from, row, col] = 0
                        elif delta > center_radius:
                            field[layer_to, d_row, d_col, layer_from, row, col] = max_value / delta**power
                        else:
                            field[layer_to, d_row, d_col, layer_from, row, col] = center_value

        return field

    def block_region(self, region: Tuple[slice]) -> None:
        self._S[(slice(None), *region)] = 0

    def update(self, dt: float, thalamic_input: np.ndarray, subcycle: int = 2) -> np.ndarray:
        spiking_fired = self._v >= 30

        # reset SNN neurons that spiked and compute their output current towards the other neurons
        zs = np.zeros((2, *self.shape))
        for _i in np.argwhere(spiking_fired):
            i = tuple(_i)
            self._v[i] = self._c[i]
            self._u[i] += self._d[i]
            zs += self._S[(slice(None), )*3 + i]

        total_current = np.maximum(thalamic_input + zs, 0)

        for _ in range(subcycle):
            v_fired = self._v >= 30
            self._v = np.where(v_fired, self._v, self._v + dt * (((0.04 * self._v**2) + (5*self._v) + 140 - self._u + total_current) / subcycle))
            self._u = np.where(v_fired, self._u, self._u + dt * self._a * ((self._b*self._v) - self._u) / subcycle)

        return spiking_fired

    @property
    def v(self):
        return self._v
