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


import os
import matplotlib.pyplot as plt

from utils import save_model


class ModelLogger:
    def __init__(self, path):
        self.params_path = path + "/params"
        self.reconstruction_path = path + "/reconstructions"
        self.checkpoints_path = path + "/checkpoints"

        if not os.path.exists(self.params_path):
            os.mkdir(self.params_path)

        if not os.path.exists(self.reconstruction_path):
            os.mkdir(self.reconstruction_path)

        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

    def log(self, smoe, checkpoint_iter=100):
        iter_ = smoe.get_iter()
        reconstruction = smoe.get_reconstruction()

        save_model(smoe, self.params_path + "/{0:08d}_params.pkl".format(iter_), best=False, reduce=True)

        plt.imsave(self.reconstruction_path + "/{0:08d}_reconstruction.png".format(iter_), reconstruction,
                   cmap='gray', vmin=0, vmax=1)

        if iter_ % checkpoint_iter == 0:
            smoe.checkpoint(self.checkpoints_path + "/{0:08d}_model.ckpt".format(iter_))
