"""
Attractor Network for 2DoF Robot Arm

Author: Henry Powell and Mathias Winkel
"""
import sys
import numpy as np

from graphics import Graphics
from ContinuousAttractorLayer import ContinuousAttractorLayer
from WavePropagationLayer import WavePropagationLayer
from setups import SETUPS

if len(sys.argv) > 1:
    selected_setup = sys.argv[1]
else:
    selected_setup = 's_maze'

try:
    setup = SETUPS[selected_setup]
except KeyError as e:
    raise ValueError('Selected setup "{}" does not exist. Chose one of \n\t{}'.format(selected_setup, '\n\t'.join(SETUPS.keys()))) from e

J = 12    # continuous attractor synaptic connection strength
T = 0.05  # continuous attractor Gaussian shift
σ = 0.03  # continuous attractor Gaussian width
τ = 0.8   # continuous attractor stabilization strength
R = 12    # continuous attractor movement recovery period

I = 25  # external DC current to stimulate selected wave propagation layer neurons
dt = 1  # simulation timestep

shape = setup['size']

wave_propagation_layer = WavePropagationLayer(shape, setup['randomize_neurons'], setup['randomize_synapses'])
continuous_attractor_layer = ContinuousAttractorLayer(shape, J, T, σ, τ)
graphics = Graphics(shape, selected_setup, setup['blocked'], setup['target_neurons'])

for region in setup['blocked']:
    continuous_attractor_layer.block_region(region)
    wave_propagation_layer.block_region(region)

continuous_attractor_layer.set_activation(setup['start_neuron'])

Δ = np.array([0, 0])
thalamic_input = np.zeros((2, *shape))

direc_update_delay = 0

coords = np.asarray(np.meshgrid(range(shape[0]), range(shape[1]))).T

for t in range(setup['t_max']):
    # random thalamic input if requested
    if setup['thalamic_input']:
        thalamic_input = np.random.uniform(0, 1, (2, *shape))

    # external drive
    for target_neuron in setup['target_neurons']:
        thalamic_input[(0, *target_neuron)] = I

    continuous_attractor_layer.update(Δ / np.asarray(shape))

    spiking_fired = wave_propagation_layer.update(dt, thalamic_input)

    place_cell_peak = continuous_attractor_layer.peak

    # layer interaction - compute direction vector
    Δ = np.array([0, 0])
    if direc_update_delay <= 0:
        overlap = continuous_attractor_layer.A * spiking_fired[0]
        total = np.sum(overlap)

        if total > 0:
            distance = coords - place_cell_peak[np.newaxis, np.newaxis, :]
            Δ = np.sum(distance * overlap[..., np.newaxis], axis=(0, 1)) / total
            direc_update_delay = R
    else:
        direc_update_delay -= dt

    if not graphics.update(t, place_cell_peak, Δ, spiking_fired, wave_propagation_layer.v, continuous_attractor_layer.A, overlap):
        print('Figure closed. Finalizing simulation.')
        break

graphics.save_video(fps=8, keep_frame_images=False)
