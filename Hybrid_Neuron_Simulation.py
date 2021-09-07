"""
Attractor Network for 2DoF Robot Arm

Author: Henry Powell and Mathias Winkel
"""
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.animation import FFMPEGVideo, ImageStack
from setups import SETUPS

if len(sys.argv) > 1:
    selected_setup = sys.argv[1]
else:
    selected_setup = 's_maze'

try:
    setup = SETUPS[selected_setup]
except KeyError as e:
    raise ValueError('Selected setup "{}" does not exist. Chose one of \n\t{}'.format(selected_setup, '\n\t'.join(SETUPS.keys()))) from e

place_cell_x = setup['size']
place_cell_y = setup['size']

J = 12  # 2.5
T = 0.05  # 0.05
sigma = 0.03  # 0.08

tau = 0.8


def update_place_cell_synapses(x: np.ndarray = None,
                               place_cell_synapses: np.ndarray = None):

    for i in range(place_cell_synapses.shape[0]):

        iidx = np.unravel_index(i, (place_cell_x, place_cell_y))
        ix = iidx[0]
        iy = iidx[1]
        ci_x = (ix - 0.5) / place_cell_x
        ci_y = (iy - 0.5) / place_cell_y
        ci = np.array([ci_x, ci_y])

        jidx = np.unravel_index(np.arange(place_cell_x*place_cell_y),
                                (place_cell_x, place_cell_y))
        jx = jidx[0]
        jy = jidx[1]
        cj_x = (jx - 0.5) / place_cell_x
        cj_y = (jy - 0.5) / place_cell_x
        cj = np.array([cj_x, cj_y]).T

        diff = ci - cj
        diff += x

        norm = np.linalg.norm(diff, axis=1)

        place_cell_synapses[i] = J * np.exp(-(norm**2/sigma**2)) - T

    return place_cell_synapses.T


def update_place_cell_activations(place_cell_synapses: np.ndarray = None,
                                  place_cell_activations: np.ndarray = None):

    place_cell_activations = place_cell_activations.flatten()
    B = np.dot(place_cell_activations, place_cell_synapses)

    summed_activations = np.sum(place_cell_activations)

    if summed_activations > 0:
        place_cell_activations = (1 - tau) * B + (tau * (B/summed_activations))
    else:
        place_cell_activations = np.zeros_like(B)

    place_cell_activations[place_cell_activations < 0] = 0

    return place_cell_activations.reshape(place_cell_x, place_cell_y)


def imprint_circular_kernel(field: np.array, layer_from: int, layer_to: int, radius: int, max_value: float,
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


# Construct spiking neuron layer
size = setup['size']
ne = size * size
ni = ne
n_neurons = ne + ni

if setup['randomize_neurons']:
    # make results reproducible
    np.random.seed(7)
    # excitatory neurons
    re = np.random.uniform(0, 1, (ne, 1))
    # inhibitory neurons
    ri = np.random.uniform(0, 1, (ne, 1))
else:
    re = np.zeros((ne, 1))
    ri = np.ones((ni, 1))

a = np.append(np.array([0.02 * np.ones((ne, 1))]),                # time scale of the recovery variable u
              np.array([0.02 + 0.08 * ri]))[:, np.newaxis]

b = np.append(np.array([0.2 * np.ones((ne, 1))]),                 # sensitivity of the recovery variable u to the subthreshold
              np.array([0.25 - 0.05 * ri]))[:, np.newaxis]        # fluctuations of the membrane potential v.

c = np.append(np.array([-65 + 15 * re**2]),                       # after-spike reset value of the membrane
              np.array([-65 * np.ones((ni, 1))]))[:, np.newaxis]  # potential v caused by fast high-threshold K+ conductances.

d = np.append(np.array([8 - 6 * re**2]),                          # after-spike reset of the recovery variable
              np.array([2*np.ones((ni, 1))]))[:, np.newaxis]      # u caused by slow high-threshold Na+ and K+ conductances.

a = a.reshape((2, size, size))
b = b.reshape((2, size, size))
c = c.reshape((2, size, size))
d = d.reshape((2, size, size))

# construct S
S = np.zeros((2, size, size) * 2)

suppression_range = 1
excitation_range = 2

# excitation layer
S = imprint_circular_kernel(S, layer_from=0, layer_to=0, radius=excitation_range, max_value=1, center_value=0)
# inhibition layer
S = imprint_circular_kernel(S, layer_from=0, layer_to=1, radius=suppression_range, max_value=0.5, center_value=0)
# inhibition deactivates a local cluster
S = imprint_circular_kernel(S, layer_from=1, layer_to=0, radius=suppression_range, max_value=-9, center_value=-9)
# some rescaling as connections between nodes are sparse
S *= 50

PC_inactive = np.ones((size, size))

for region in setup['blocked']:
    S[(slice(None), *region)] = 0
    PC_inactive[region] = 0
target_neuron = setup['target_neuron']

v = -65 * np.ones((ne+ni, 1)).reshape((2, size, size))
u = b*v
firings = np.empty((0, 2))

n = 1

# Construct continuous attractor layer
place_cell_synapses = np.zeros((place_cell_x * place_cell_y, place_cell_x * place_cell_y))

direc = np.array([10, 10]) / np.array([place_cell_x, place_cell_y])

place_cell_activations = np.zeros((place_cell_x, place_cell_y))
place_cell_activations[setup['start_neuron']] = 1.

trajectory = np.zeros((size, size))

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
        ax.myarrow = ax.annotate("",
                                 xy=(setup['target_neuron'][0]+.5, setup['target_neuron'][1]+.5),
                                 xytext=(-20, 50), textcoords='offset pixels',
                                 arrowprops=dict(arrowstyle="->"))

    if hasattr(ax, 'myhatch'):
        for coll in ax.myhatch.collections:
            ax.collections.remove(coll)
    ax.myhatch = ax.contourf(mask, 1, hatches=['', '////'], alpha=0)


max_plot = list()

for t in range(setup['t_max']):
    if setup['thalamic_input']:
        I = np.random.randn(n_neurons, 1).reshape(2, size, size)
    else:
        I = np.zeros((n_neurons, 1)).reshape(2, size, size)

    # external drive
    I[(0, *target_neuron)] = 25

    place_cell_synapses = update_place_cell_synapses(direc, place_cell_synapses)

    place_cell_activations = update_place_cell_activations(place_cell_synapses,
                                                           place_cell_activations)
    place_cell_activations = np.multiply(place_cell_activations, PC_inactive)

    # normalize place cell activations to prevent
    place_cell_activations /= np.max(place_cell_activations)

    spiking_fired = v >= 30
    spiking_fired_excite = np.where(v[:ne] >= 30)[0]

    fire_grid = 1. * spiking_fired[0]
    # fire_grid[target_neuron] = 2.0

    overlap = np.multiply(place_cell_activations, fire_grid)
    overlap_coordinates = np.argwhere(overlap > 0)

    if len(overlap_coordinates) == 0:
        direc = np.array([0, 0])
    else:
        place_cell_peak = np.unravel_index(np.argmax(place_cell_activations), place_cell_activations.shape)
        average_spiking_pos = np.mean(overlap_coordinates, axis=0)
        vec = place_cell_peak - average_spiking_pos
        norm = np.linalg.norm(vec)
        if norm == 0:
            direc = np.array([0, 0])
        else:
            direc = vec / np.linalg.norm(vec)
            direc = direc / np.array([place_cell_x, place_cell_y])

        trajectory *= 0.95
        trajectory[tuple(np.round(average_spiking_pos).astype(int))] = 1.
        trajectory[place_cell_peak] = -1.

    zs = np.zeros((2, size, size))

    # reset SNN neurons that spiked and compute their output current towards the other neurons
    for _i in np.argwhere(spiking_fired):
        i = tuple(_i)
        v[i] = c[i]
        u[i] += d[i]
        zs += S[(slice(None), )*3 + i]

    total_current = np.maximum(I + zs, 0)

    # suppress SNN neuron activity in overlap region with place cells
    suppression_range = 5
    for i in overlap_coordinates:
        total_current[0, i[0]-suppression_range:i[0]+suppression_range+1, i[1]-suppression_range:i[1]+suppression_range+1] = 0

    subcycle = 2
    for update in range(subcycle):
        v_fired = v >= 30
        v = np.where(v_fired, v, v + (((0.04 * v**2) + (5*v) + 140 - u + total_current) / subcycle))
        u = np.where(v_fired, u, u + a * ((b*v) - u) / subcycle)

    if t % 10 == 0:
        # ########### Plots for animation ###################
        fig_vid.suptitle(f't = {t}ms')
        imupdate(ax_vid[0, 0], fire_grid, vmin=0, vmax=2)
        imupdate(ax_vid[0, 1], v[0], vmin=-70, vmax=30)
        imupdate(ax_vid[0, 2], 1 * spiking_fired[1], vmin=0, vmax=2)
        imupdate(ax_vid[0, 3], v[1], vmin=-70, vmax=30)
        imupdate(ax_vid[1, 0], place_cell_activations)
        imupdate(ax_vid[1, 1], overlap)
        imupdate(ax_vid[1, 2], trajectory, vmin=-1, vmax=1, cmap='bwr')

        for ax in ax_vid.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig_vid.tight_layout()

        # ########### Plots for publication ###################
        fig_pub.suptitle(f't = {t}ms', fontsize=24)
        imupdate(ax_pub, place_cell_activations, cmap='Greys', overlay=fire_grid)

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
        animation.add_frame(fig_vid)  # comment to prevent saving plots to disc
        pub_images.add_frame(fig_pub)
        pub_images2.add_frame(fig_pub2)

    if not plt.fignum_exists(fig_vid.number):
        print('Figure closed. Finalizing simulation.')
        break

animation.save(selected_setup, fps=4, keep_frame_images=False)
