import numpy as np
from typing import Tuple


class ContinuousAttractorLayer:

    def __init__(self, shape: Tuple[int], J: float, T: float, σ: float, τ: float):
        self.shape = shape
        self._J = J
        self._T = T
        self._σ = σ
        self._τ = τ

        # real-space position of the place cell activations
        ci = np.asarray(np.meshgrid(
            (np.arange(self.shape[0]) - 0.5) / self.shape[0],
            (np.arange(self.shape[1]) - 0.5) / self.shape[1])).T

        # precompute pairwise difference between all entries in ci for the place_cell_synapses
        self._ci_diff = ci[:, :, np.newaxis, np.newaxis, :] - ci[np.newaxis, np.newaxis, :, :, :]

        self._place_cell_synapses = np.zeros(self.shape * 2)
        self._place_cell_activations = np.zeros(self.shape)
        self._place_cell_blocked = np.ones(self.shape)

        # cache place cell synapses for Δ == (0, 0)
        self._place_cell_synapses_0 = None
        self._update_place_cell_synapses(np.array([0, 0]))
        self._place_cell_synapses_0 = self._place_cell_synapses

    def block_region(self, region: Tuple[slice]) -> None:
        self._place_cell_blocked[region] = 0

    def set_activation(self, point: Tuple[int]) -> None:
        if point is not None:
            self._place_cell_activations[point] = 1.

    def _update_place_cell_synapses(self, Δ: np.ndarray) -> None:
        if Δ[0] == 0 and Δ[1] == 0 and self._place_cell_synapses_0 is not None:
            self._place_cell_synapses = self._place_cell_synapses_0
        else:
            diff = self._ci_diff + Δ
            np.square(diff, out=diff)
            # self._place_cell_synapses = J * np.exp(-(norm_sq/σ**2)) - T
            self._place_cell_synapses = np.sum(diff, axis=-1)
            self._place_cell_synapses /= -self._σ**2
            np.exp(self._place_cell_synapses, out=self._place_cell_synapses)
            self._place_cell_synapses *= self._J
            self._place_cell_synapses -= self._T

    def _update_place_cell_activations(self) -> None:

        Σ = np.sum(self._place_cell_activations)

        if Σ > 0:
            B = np.einsum('ij,ijkl->kl', self._place_cell_activations, self._place_cell_synapses)
            self._place_cell_activations = (1 - self._τ) * B + self._τ/Σ * B
            self._place_cell_activations[self._place_cell_activations < 0] = 0
            self._place_cell_activations *= self._place_cell_blocked
            self._place_cell_activations /= self._place_cell_activations.max()

    def update(self, Δ: np.ndarray) -> np.ndarray:
        self._update_place_cell_synapses(Δ)
        self._update_place_cell_activations()
        return self.peak

    @property
    def A(self) -> np.ndarray:
        return self._place_cell_activations

    @property
    def peak(self) -> np.ndarray:
        return np.asarray(np.unravel_index(np.argmax(self.A), self.A.shape))
