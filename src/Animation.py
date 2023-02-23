"""
Code taken from:
https://github.com/Vemundss/unitary_memory_indexing/blob/main/src/AnimatedScatter.py
"""
# import matplotlib
# matplotlib.use('nbAgg') # Tried to use same backend as jupyter notebook
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np


def projection_rejection(u, v):
    """
    projection of u on v, and rejection of u from v
    """
    proj = ((u @ v) / (v @ v)) * v
    reject = u - proj
    return proj, reject


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, cluster_data, weight_data, loss_history, n_clusters=4,acc=None):
        """
        Args:
            cluster_data: (N,2), where N=4*n -> four clusters
            weight_data: (#epochs,2,4)
            acc: end accuracy
        """
        self.cluster_data = cluster_data
        self.weight_data = weight_data
        self.loss_history = loss_history
        self.n_clusters = n_clusters
        self.acc = acc

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.

        self.save_count = weight_data.shape[0]
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.setup_plot,
            interval=25,  # time in ms between frames
            # repeat_delay=1000, # delay before loop
            blit=False,  # for OSX?
            save_count=self.save_count,  # #frames
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        data = self.cluster_data
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        self.means = np.mean(
            np.reshape(data, (self.n_clusters, int(data.shape[0] / self.n_clusters), data.shape[-1])), axis=1
        )
        color_idxs = np.argmax(
            self.means @ self.weight_data[0], axis=0
        )  # shapes (self.n_clusters,2) @ (2,self.n_clusters)
        cluster_N = int(data.shape[0] / self.n_clusters)
        self.weight_arrows = []
        self.rejection_arrows = []
        self.projection_arrows = []
        for color, mean, i in zip(colors, self.means, range(len(colors))):
            self.ax.scatter(
                data[i * cluster_N : (i + 1) * cluster_N, 0],
                data[i * cluster_N : (i + 1) * cluster_N, 1],
                color=color,
            )
            self.ax.arrow(
                0,
                0,
                *mean,
                length_includes_head=True,
                width=0.01,
                color=(0, 0, 0, 0.5),  # semi-transparent black arrow
            )

            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
            )

            proj, reject = projection_rejection(self.weight_data[0, :, i], mean)
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)
        self.ax.grid("on")
        self.ax.set_title("Loss={}".format(self.loss_history[0]))
        # self.ax.set_xlim([-1.5,1.5])
        # self.ax.set_ylim([-1.5,1.5])

    def update(self, k):
        """Update the scatter plot."""
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        color_idxs = np.argmax(
            self.means @ self.weight_data[k], axis=0
        )  # shapes (4,2) @ (2,4)
        for i in range(self.n_clusters):
            self.weight_arrows.pop(0).remove()  # delete arrow
            self.rejection_arrows.pop(0).remove()
            self.projection_arrows.pop(0).remove()
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],  # color=(1, 0, 0, 0.5),
            )

            proj, reject = projection_rejection(
                self.weight_data[k, :, i], self.means[i]
            )
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.5,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)

        # self.ax.set_xlim([-1.5,1.5])
        # self.ax.set_ylim([-1.5,1.5])
        self.ax.set_title(f"Loss={self.loss_history[k]} End accuracy={self.acc}")

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.weight_arrows


class AnimatedScatter_GradientData(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, cluster_data, weight_data, jacobigrad, crossgrad, loss_history, n_clusters=4,acc=None):
        """
        Args:
            cluster_data: (N,2), where N=4*n -> four clusters
            weight_data: (#epochs,2,4)
            acc: end accuracy
        """
        self.cluster_data = cluster_data
        self.weight_data = weight_data
        self.loss_history = loss_history
        self.n_clusters = n_clusters
        self.acc = acc
        self.jacobigrad = jacobigrad
        self.crossgrad = crossgrad

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.

        self.save_count = weight_data.shape[0]
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.setup_plot,
            interval=25,  # time in ms between frames
            # repeat_delay=1000, # delay before loop
            blit=False,  # for OSX?
            save_count=self.save_count,  # #frames
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        data = self.cluster_data
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        self.means = np.mean(
            np.reshape(data, (self.n_clusters, int(data.shape[0] / self.n_clusters), data.shape[-1])), axis=1
        )
        color_idxs = np.argmax(
            self.means @ self.weight_data[0], axis=0
        )  # shapes (self.n_clusters,2) @ (2,self.n_clusters)
        cluster_N = int(data.shape[0] / self.n_clusters)
        self.weight_arrows = []
        self.rejection_arrows = []
        self.projection_arrows = []
        self.jacobi_arrows = []
        self.cross_arrows = []
        for color, mean, i in zip(colors, self.means, range(len(colors))):
            self.ax.scatter(
                data[i * cluster_N : (i + 1) * cluster_N, 0],
                data[i * cluster_N : (i + 1) * cluster_N, 1],
                color=color,
            )
            self.ax.arrow(
                0,
                0,
                *mean,
                length_includes_head=True,
                width=0.01,
                color=(0, 0, 0, 0.5),  # semi-transparent black arrow
            )

            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
            )

            jacobi_arrow = self.ax.arrow(
                *self.weight_data[0, :, i],
                *self.jacobigrad[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=(1, 0, 0, 0.5),  # semi-transparent red arrow
                label="Jacobian gradient",
            )

            cross_arrow = self.ax.arrow(
                *self.weight_data[0, :, i],
                *self.crossgrad[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=(0, 0, 1, 0.5),  # semi-transparent blue arrow
                label="Cross entropy gradient",
            )


            proj, reject = projection_rejection(self.weight_data[0, :, i], mean)
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)
            self.jacobi_arrows.append(jacobi_arrow)
            self.cross_arrows.append(cross_arrow)
        self.ax.grid("on")
        self.ax.set_xlim([-1.5,1.5])
        self.ax.set_ylim([-1.5,1.5])
        plt.legend()
        self.ax.set_title("Loss={}".format(self.loss_history[0]))

    def update(self, k):
        """Update the scatter plot."""
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        color_idxs = np.argmax(
            self.means @ self.weight_data[k], axis=0
        )  # shapes (4,2) @ (2,4)
        for i in range(self.n_clusters):
            self.weight_arrows.pop(0).remove()  # delete arrow
            self.rejection_arrows.pop(0).remove()
            self.projection_arrows.pop(0).remove()
            self.jacobi_arrows.pop(0).remove()
            self.cross_arrows.pop(0).remove()
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],  # color=(1, 0, 0, 0.5),
            )

            jacobi_arrow = self.ax.arrow(
                *self.weight_data[k, :, i],
                *self.jacobigrad[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=(1, 0, 0, 0.5),  # semi-transparent red arrow
            )

            cross_arrow = self.ax.arrow(
                *self.weight_data[k, :, i],
                *self.crossgrad[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=(0, 0, 1, 0.5),  # semi-transparent blue arrow
            )

            proj, reject = projection_rejection(
                self.weight_data[k, :, i], self.means[i]
            )
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.5,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[
                    color_idxs[i]
                ],  # rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)
            self.jacobi_arrows.append(jacobi_arrow)
            self.cross_arrows.append(cross_arrow)

        self.ax.set_title(f"Loss={self.loss_history[k]} End accuracy={self.acc}")
        self.ax.set_xlim([-1.5,1.5])
        self.ax.set_ylim([-1.5,1.5])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.weight_arrows