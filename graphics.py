import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import copy
from typing import Tuple

from utils.animation import FFMPEGVideo, ImageStack


class Graphics:
    def __init__(self, shape: Tuple[int], name: str, blocked_neurons, target_neurons):
        self._blocked_neurons = blocked_neurons
        self._target_neurons = target_neurons
        self.trajectory = np.zeros(shape)

        plt.ion()
        self.fig_vid, self.ax_vid = plt.subplots(nrows=2, ncols=4, squeeze=True, figsize=(10, 6))
        self.fig_vid.set_tight_layout(True)
        self.ax_vid[0, 0].set_title('Exci. Firing Pattern', fontsize=8)
        self.ax_vid[0, 1].set_title('Exci. SNN Membrane Potential', fontsize=8)
        self.ax_vid[0, 2].set_title('Inhi. Firing Pattern', fontsize=8)
        self.ax_vid[0, 3].set_title('Inhi. SNN Membrane Potential', fontsize=8)
        self.ax_vid[1, 0].set_title('Place Cell Activations', fontsize=8)
        self.ax_vid[1, 1].set_title('Overlap', fontsize=8)
        self.ax_vid[1, 2].set_title('Trajectory', fontsize=8)
        self.ax_vid[1, 3].remove()

        self.fig_pub, self.ax_pub = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(3, 3.25))
        self.fig_pub.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.)

        self.fig_pub2, self.ax_pub2 = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(6, 3))
        self.fig_pub2.set_tight_layout(True)
        self.ax_pub2[0].set_title('Excitatory Firing Pattern', fontsize=14)
        self.ax_pub2[1].set_title('Inhibitory Firing Pattern', fontsize=14)

        self.animation = FFMPEGVideo(name)
        self.animation_pub = FFMPEGVideo(name + '_pub')
        self.pub_images = ImageStack(name)
        self.pub_images2 = ImageStack(name + '.exci_inhi')

        self.my_cmap = copy.copy(plt.cm.get_cmap('gray'))  # get a copy of the gray color map
        self.my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values

        self.mask_cmap = ListedColormap([(0, 0, 0, 0), (0.3, 0.3, 0.3, 1.)])

    def _imupdate(self, ax, data, overlay=None, *args, **kwargs):

        # mask geometry to make the setup visible in the plot
        mask = np.zeros_like(data)
        data_plot = data.copy().astype(float)
        for region in self._blocked_neurons:
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
            ax.myplot = ax.imshow(data_plot, *args, **kwargs, zorder=1)
            if overlay is not None:
                ax.myoverlay = ax.imshow(overlay_tmp, vmin=0, vmax=2, cmap=self.my_cmap, zorder=1)

            ax.decoration = [
                ax.vlines(np.arange(*ax.get_xlim(), 1), *ax.get_ylim(), colors='w', linestyles='-', linewidth=0.25, zorder=2),
                ax.hlines(np.arange(*ax.get_ylim(), -1), *ax.get_xlim(), colors='w', linestyles='-', linewidth=0.25, zorder=2),
                ax.imshow(mask, alpha=1., cmap=self.mask_cmap, zorder=3),
                ax.contourf(mask, 1, hatches=['', 'xx'], alpha=0, zorder=4),
                [ax.annotate("",
                             xy=(target_neuron[0]+.25, target_neuron[1]+.75),
                             xytext=(-20, 50), textcoords='offset pixels',
                             zorder=4,
                             arrowprops=dict(arrowstyle="->")) for target_neuron in self._target_neurons]
             ]

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])


    def update(self, t, place_cell_peak, Δ, spiking_fired, membrane_potential, attractor_activity, overlap):
        # record trajectory for plotting
        self.trajectory *= 0.99
        self.trajectory[tuple(np.round(place_cell_peak + Δ).astype(int))] = 1.  # if direc == 0 this will be overwritten by the next line
        self.trajectory[tuple(place_cell_peak)] = -1.

        fire_grid = 1. * spiking_fired[0]

        # ########### Plots for animation ###################
        self.fig_vid.suptitle(f't = {t}ms')
        self._imupdate(self.ax_vid[0, 0], fire_grid, vmin=0, vmax=2)
        self._imupdate(self.ax_vid[0, 1], membrane_potential[0], vmin=-70, vmax=30)
        self._imupdate(self.ax_vid[0, 2], 1 * spiking_fired[1], vmin=0, vmax=2)
        self._imupdate(self.ax_vid[0, 3], membrane_potential[1], vmin=-70, vmax=30)
        self._imupdate(self.ax_vid[1, 0], attractor_activity)
        self._imupdate(self.ax_vid[1, 1], overlap)
        self._imupdate(self.ax_vid[1, 2], self.trajectory, vmin=-1, vmax=1, cmap='bwr')

        # ########### Plots for publication ###################
        self.fig_pub.suptitle(f't = {t}ms', fontsize=24)
        self._imupdate(self.ax_pub, attractor_activity, cmap='Greys', overlay=fire_grid)

        self._imupdate(self.ax_pub2[0], fire_grid, vmin=0, vmax=2, cmap='Greys')
        self._imupdate(self.ax_pub2[1], 1 * spiking_fired[1], vmin=0, vmax=2, cmap='Greys')

        plt.show()

        plt.pause(0.1)
        self.animation.add_frame(self.fig_vid)
        self.animation_pub.add_frame(self.fig_pub)
        self.pub_images.add_frame(self.fig_pub)
        self.pub_images2.add_frame(self.fig_pub2)

        return plt.fignum_exists(self.fig_vid.number)

    def save_video(self, fps: int = 8, keep_frame_images=False):
        self.animation.save(fps=fps, keep_frame_images=keep_frame_images)
        self.animation_pub.save(fps=fps, keep_frame_images=keep_frame_images)
