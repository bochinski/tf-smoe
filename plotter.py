# ---------------------------------------------------------------------
# Copyright (c) 2018 TU Berlin, Communication Systems Group
# Written by Erik Bochinski <bochinski@nue.tu-berlin.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------


from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import glob

mpl.rcParams['image.cmap'] = 'jet'


def psnr(mse):
    return 10 * np.log10(255 ** 2 / mse)


class ImagePlotter:
    def __init__(self, path=None, options=(), quiet=False):
        self.path = path
        self.options = options
        self.quiet = quiet
        self.fig = None

        if self.path is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                files = glob.glob(path + "/*")
                list(map(lambda x: os.remove(x), files))

        # TODO not nice
        num_options = len(self.options)
        if num_options <= 4:
            self.rows = 1
            self.cols = num_options
        else:
            self.rows = 2
            self.cols = num_options % 4

        # self.fig, self.axes = plt.subplots(1, self.cols)  # plt.subplots(self.rows, self.cols)

        # TODO only hotfix to display hist properly
        if "pis_hist" in self.options:
            self.fig = plt.figure()

            gs = GridSpec(2, len(options) - 1)  # 2 rows, 3 columns

            self.axes = []
            for i in range(len(options) - 1):
                self.axes.append(self.fig.add_subplot(gs[0, i]))
            self.axes.append(self.fig.add_subplot(gs[1, :]))


        else:
            self.fig, self.axes = plt.subplots(1, self.cols)

        if not self.quiet:
            self.fig.show()
            self.fig.canvas.draw()

    def plot(self, smoe):
        for idx, option in enumerate(self.options):
            row = int(idx / self.cols)
            col = int(idx % self.cols)

            # TODO hist hotfix
            ax = self.axes[col]
            """
            # no row, col indexes if there is only one row
            if len(self.axes) == len(self.options):
                ax = self.axes[col]
            else:
                ax = self.axes[row, col]
            """
            ax.clear()

            if option == "orig":
                ax.imshow(smoe.get_original_image(), cmap='gray', interpolation='None', vmin=0, vmax=1)
                ax.set_title("Original")
            elif option == "reconstruction":
                ax.imshow(smoe.get_reconstruction(), cmap='gray', interpolation='None', vmin=0, vmax=1)
                ax.set_title("Reconstruction")
            elif option == "gating":
                w_e_opt = smoe.get_weight_matrix_argmax()
                ax.imshow(w_e_opt, interpolation='None', cmap='prism')
                ax.set_title("Gating")
            elif option == "pis_hist":
                # TODO hist hotfix
                ax = self.axes[-1]
                params = smoe.get_params()
                pis_pos_idx = params['pis'] > 0
                ax.hist(params['pis'][pis_pos_idx], 500)
                used = np.count_nonzero(pis_pos_idx)
                total = params['pis'].shape[0]
                ax.set_title('Histogram of pis {0:d} / {1:d} ({2:.2f})'.format(used, total, 100. * used / total))

        iters_loss, losses = zip(*smoe.get_losses())
        iters_mse, mses = zip(*smoe.get_mses())
        assert iters_loss == iters_mse, "mse/loss logging out of sync"
        self.fig.suptitle(
            'start, best, last: {0:.6f} / {1:.6f} / {2:.6f}\n'
            'MSE: start, best, last: {3:.2f} / {4:.2f} / {5:.2f}\n'
            'PSNR: start, best, last: {6:.2f} / {7:.2f} / {8:.2f}'.format(losses[0],
                                                                          smoe.get_best_loss(),
                                                                          losses[-1],
                                                                          mses[0],
                                                                          smoe.get_best_mse(),
                                                                          mses[-1],
                                                                          psnr(mses[0]),
                                                                          psnr(smoe.get_best_mse()),
                                                                          psnr(mses[-1])),
            y=1.)

        if not self.quiet:
            self.fig.canvas.draw()

        if self.path:
            name = "/{0:08d}.png".format(smoe.get_iter())
            self.fig.savefig(self.path + "/" + name, dpi=600)

    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)


class LossPlotter:
    def __init__(self, path=None, quiet=False):
        self.path = path
        self.quiet = quiet
        self.fig = None

        self.fig = plt.figure()
        self.ax_loss = self.fig.add_subplot(111)
        self.ax_mse = self.ax_loss.twinx()
        self.ax_pis = self.ax_loss.twinx()

        self.ax_loss.set_ylabel('loss', color='b')
        self.ax_loss.tick_params('y', colors='b')
        self.ax_mse.set_ylabel('MSE', color='r')
        self.ax_mse.tick_params('y', colors='r')
        self.ax_pis.set_ylabel('MSE', color='gray')
        self.ax_pis.tick_params('y', colors='gray')

        # if self.path is not None:
        #    if not os.path.exists(path):
        #        os.mkdir(path)
        #    else:
        #        files = glob.glob(path + "/*")
        #        list(map(lambda x: os.remove(x), files))


        if not self.quiet:
            self.fig.show()
            self.fig.canvas.draw()

    def plot(self, smoe):
        self.ax_loss.clear()
        self.ax_mse.clear()
        self.ax_pis.clear()

        self.ax_pis.spines['right'].set_position(('outward', 50))

        iters_loss, losses = zip(*smoe.get_losses())
        iters_mse, mses = zip(*smoe.get_mses())
        iters_pis, pis = zip(*smoe.get_num_pis())
        assert iters_loss == iters_mse and iters_mse == iters_pis, \
            "mse/loss logging out of sync" + str((iters_loss, iters_mse, iters_pis))

        self.ax_loss.clear()
        self.ax_loss.set_ylim(top=np.mean(losses[-100:])+losses[-1]/2)
        self.ax_mse.set_ylim(top=np.mean(mses[-100:])++mses[-1]/2)

        self.ax_loss.set_title(
            'start, best, last: {0:.6f} / {1:.6f} / {2:.6f}\n'
            'MSE: start, best, last: {3:.2f} / {4:.2f} / {5:.2f}\n'
            'PSNR: start, best, last: {6:.2f} / {7:.2f} / {8:.2f}'.format(losses[0],
                                                                          smoe.get_best_loss(),
                                                                          losses[-1],
                                                                          mses[0],
                                                                          smoe.get_best_mse(),
                                                                          mses[-1],
                                                                          psnr(mses[0]),
                                                                          psnr(smoe.get_best_mse()),
                                                                          psnr(mses[-1]))
        )
        self.ax_loss.plot(iters_loss, losses, color='b')
        self.ax_mse.plot(iters_mse, mses, color='r')
        self.ax_pis.plot(iters_pis, pis, color='gray')

        if self.path:
            self.fig.savefig(self.path, bbox_inches='tight')

        if not self.quiet:
            self.fig.canvas.draw()

    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)


class DenoisePlotter:
    def __init__(self, y, z, ref, path=None):
        self.path = path
        if self.path and not os.path.exists(self.path):
            os.mkdir(self.path)

        self.y = y
        self.z = z
        self.ref = ref
        self.psnrs = []

        self.fig = plt.figure(figsize=(12, 7))

        self.num_plots = 4
        gs = GridSpec(2, self.num_plots)

        self.axes = []
        for i in range(self.num_plots):
            self.axes.append(self.fig.add_subplot(gs[0, i]))

        self.axes.append(self.fig.add_subplot(gs[1, :]))

        self.axes[0].set_title("original")
        self.axes[0].imshow(self.y, cmap='gray', interpolation='None', vmin=0, vmax=1)

        y_est = self.ref
        mse = mean_squared_error(y_est * 255, self.y * 255)
        psnr = 10 * np.log10(255 ** 2 / mse)
        ssim = compare_ssim(y_est, self.y, data_range=1)

        self.axes[2].set_title('reference \n mse: '+str(round(mse,2))+' psnr '+str(round(psnr,2))+'\n ssim: '+str(round(ssim,3)))
        self.axes[2].imshow(y_est, cmap='gray', interpolation='None', vmin=0, vmax=1)

        y_est = self.z
        mse = mean_squared_error(y_est * 255, self.y * 255)
        psnr = 10 * np.log10(255 ** 2 / mse)
        ssim = compare_ssim(y_est, self.y, data_range=1)

        self.axes[3].set_title(
            'noisy input \n mse: ' + str(round(mse, 2)) + ' psnr ' + str(round(psnr, 2)) + '\n ssim: ' + str(
                round(ssim, 3)))
        self.axes[3].imshow(self.z, cmap='gray', interpolation='None', vmin=0, vmax=1)

    def plot(self, smoe):
        # for ax in self.axes:
        #    ax.clear()

        y_est = smoe.get_reconstruction()
        mse = mean_squared_error(y_est * 255, self.y * 255)
        psnr_ = psnr(mse)
        self.psnrs.append(psnr_)
        ssim = compare_ssim(y_est, self.y, data_range=1)

        self.axes[1].clear()
        self.axes[1].set_title('denoised \n mse: '+str(round(mse,2))+' psnr '+str(round(psnr_,2))+'\n ssim: '+str(round(ssim,3)))
        self.axes[1].imshow(y_est, cmap='gray', interpolation='None', vmin=0, vmax=1)

        self.axes[4].clear()
        self.axes[4].set_title("Max: {0:.2f}, Last: {1:.2f}".format(np.max(self.psnrs), self.psnrs[-1]))
        self.axes[4].set_ylabel("PSNR in dB")
        self.axes[4].plot(self.psnrs)

        self.fig.canvas.draw()

        if self.path:
            name = "/{0:08d}.png".format(smoe.get_iter())
            self.fig.savefig(self.path + "/" + name, dpi=600)
