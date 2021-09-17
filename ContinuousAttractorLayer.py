import numpy as np
from typing import Tuple


class ContinuousAttractorLayer:

    def __init__(self, shape: Tuple[int]):
        self.shape = shape

        # real-space position of the place cell activations
        ci = np.asarray(np.meshgrid(
            (np.arange(self.shape[0]) - 0.5) / self.shape[0],
            (np.arange(self.shape[1]) - 0.5) / self.shape[1])).T

        # precompute pairwise difference between all entries in ci for the place_cell_synapses
        self._ci_diff = ci[:, :, np.newaxis, np.newaxis, :] - ci[np.newaxis, np.newaxis, :, :, :]

        self._place_cell_synapses = np.zeros(self.shape * 2)
        self._place_cell_activations = np.zeros(self.shape)
        self._place_cell_blocked = np.ones(self.shape)

    def block_region(self, region: Tuple[slice]) -> None:
        self._place_cell_blocked[region] = 0

    def set_activation(self, point: Tuple[int]) -> None:
        self._place_cell_activations[point] = 1.

    def _update_place_cell_synapses(self, Δ: np.ndarray, J: float, T: float, σ: float) -> None:
        diff = self._ci_diff + Δ
        norm_sq = np.sum(np.square(diff, out=diff), axis=-1)
        self._place_cell_synapses = J * np.exp(-(norm_sq/σ**2)) - T

    def _update_place_cell_activations(self, τ: float) -> None:

        Σ = np.sum(self._place_cell_activations)

        if Σ > 0:
            B = np.einsum('ij,ijkl->kl', self._place_cell_activations, self._place_cell_synapses)
            self._place_cell_activations = (1 - τ) * B + τ/Σ * B
            self._place_cell_activations[self._place_cell_activations < 0] = 0

    def update(self, Δ: np.ndarray, J: float, T: float, σ: float, τ: float):
        self._update_place_cell_synapses(Δ, J, T, σ)
        self._update_place_cell_activations(τ)
        self._place_cell_activations *= self._place_cell_blocked
        self._place_cell_activations /= self._place_cell_activations.max()

    @property
    def A(self) -> np.ndarray:
        return self._place_cell_activations

    @property
    def peak(self) -> np.ndarray:
        return np.asarray(np.unravel_index(np.argmax(self.A), self.A.shape))
