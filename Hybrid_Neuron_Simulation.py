"""
Attractor Network for 2DoF Robot Arm

Author: Henry Powell and Mathias Winkel
"""
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.animation import FFMPEGVideo, ImageStack
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

for region in setup['blocked']:
    continuous_attractor_layer.block_region(region)
    wave_propagation_layer.block_region(region)

target_neurons = setup['target_neurons']
start_neuron = setup['start_neuron']

if start_neuron is not None:
    continuous_attractor_layer.set_activation(start_neuron)

trajectory = np.zeros(shape)

plt.ion()
fig_vid, ax_vid = plt.subplots(nrows=2, ncols=4, squeeze=True, figsize=(10, 6))
ax_vid[0, 0].set_title('Exci. Firing Pattern', fontsize=8)
ax_vid[0, 1].set_title('Exci. SNN Membrane Potential', fontsize=8)
ax_vid[0, 2].set_title('Inhi. Firing Pattern', fontsize=8)
ax_vid[0, 3].set_title('Inhi. SNN Membrane Potential', fontsize=8)
ax_vid[1, 0].set_title('Place Cell Activations', fontsize=8)
ax_vid[1, 1].set_title('Overlap', fontsize=8)
ax_vid[1, 2].set_title('Trajectory', fontsize=8)
ax_vid[1, 3].remove()

fig_pub, ax_pub = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(3, 3.25))

fig_pub2, ax_pub2 = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(6, 3))
ax_pub2[0].set_title('Excitatory Firing Pattern', fontsize=14)
ax_pub2[1].set_title('Inhibitory Firing Pattern', fontsize=14)

animation = FFMPEGVideo()
pub_images = ImageStack(selected_setup)
pub_images2 = ImageStack(selected_setup + '.exci_inhi')

my_cmap = copy.copy(plt.cm.get_cmap('gray'))  # get a copy of the gray color map
my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values


def imupdate(ax, data, overlay=None, *args, **kwargs):

    # mask geometry to make the setup visible in the plot
    mask = np.zeros_like(data)
    data_plot = data.copy().astype(float)
    for region in setup['blocked']:
        mask[region] = 1
        data_plot[region] = np.nan

    if overlay is not None:
        overlay_tmp = overlay.copy()
        overlay_tmp[overlay_tmp == 0] = np.nan

    if hasattr(ax, 'myplot'):
        ax.myplot.set_data(data_plot)

        if overlay is not None:
            ax.myoverlay.set_data(overlay_tmp)
    else:
        ax.myplot = ax.imshow(data_plot, *args, **kwargs)
        if overlay is not None:
            ax.myoverlay = ax.imshow(overlay_tmp, vmin=0, vmax=2, cmap=my_cmap)
        ax.myarrow = [ax.annotate("",
                                  xy=(target_neuron[0]+.5, target_neuron[1]+.5),
                                  xytext=(-20, 50), textcoords='offset pixels',
                                  arrowprops=dict(arrowstyle="->")) for target_neuron in target_neurons]

    if hasattr(ax, 'myhatch'):
        for coll in ax.myhatch.collections:
            ax.collections.remove(coll)
    ax.myhatch = ax.contourf(mask, 1, hatches=['', '////'], alpha=0)


Δ = np.array([0, 0])
thalamic_input = np.zeros((2, *shape))

direc_update_delay = 0

coords = np.asarray(np.meshgrid(range(shape[0]), range(shape[1]))).T

for t in range(setup['t_max']):
    # random thalamic input if requested
    if setup['thalamic_input']:
        thalamic_input = np.random.uniform(0, 1, (2, *shape))

    # external drive
    for target_neuron in target_neurons:
        thalamic_input[(0, *target_neuron)] = I

    if start_neuron is not None:
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

    # record trajectory for plotting
    trajectory *= 0.99
    trajectory[tuple(np.round(place_cell_peak + Δ).astype(int))] = 1.  # if direc == 0 this will be overwritten by the next line
    trajectory[tuple(place_cell_peak)] = -1.

    if t % 1 == 0:
        fire_grid = 1. * spiking_fired[0]

        # ########### Plots for animation ###################
        fig_vid.suptitle(f't = {t}ms')
        imupdate(ax_vid[0, 0], fire_grid, vmin=0, vmax=2)
        imupdate(ax_vid[0, 1], wave_propagation_layer.v[0], vmin=-70, vmax=30)
        imupdate(ax_vid[0, 2], 1 * spiking_fired[1], vmin=0, vmax=2)
        imupdate(ax_vid[0, 3], wave_propagation_layer.v[1], vmin=-70, vmax=30)
        imupdate(ax_vid[1, 0], continuous_attractor_layer.A)
        imupdate(ax_vid[1, 1], overlap)
        imupdate(ax_vid[1, 2], trajectory, vmin=-1, vmax=1, cmap='bwr')

        for ax in ax_vid.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig_vid.tight_layout()

        # ########### Plots for publication ###################
        fig_pub.suptitle(f't = {t}ms', fontsize=24)
        imupdate(ax_pub, continuous_attractor_layer.A, cmap='Greys', overlay=fire_grid)

        for ax in [ax_pub]:
            ax.set_xticks([])
            ax.set_yticks([])

        fig_pub.tight_layout()

        imupdate(ax_pub2[0], fire_grid, vmin=0, vmax=2, cmap='Greys')
        imupdate(ax_pub2[1], 1 * spiking_fired[1], vmin=0, vmax=2, cmap='Greys')

        for ax in ax_pub2.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig_pub2.tight_layout()

        plt.show()

        plt.pause(0.1)
        animation.add_frame(fig_vid)
        pub_images.add_frame(fig_pub)
        pub_images2.add_frame(fig_pub2)

    if not plt.fignum_exists(fig_vid.number):
        print('Figure closed. Finalizing simulation.')
        break

animation.save(selected_setup, fps=8, keep_frame_images=False)
