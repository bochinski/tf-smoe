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


import pickle


def reduce_params(params):
    idx = params['pis'] > 0
    params['pis'] = params['pis'][idx]
    params['A'] = params['A'][idx]
    params['nu_e'] = params['nu_e'][idx]
    params['gamma_e'] = params['gamma_e'][idx]
    params['musX'] = params['musX'][idx]
    return params


def save_model(smoe, path, best=False, reduce=True):
    if best:
        params = smoe.get_best_params()
    else:
        params = smoe.get_params()
    if reduce:
        params = reduce_params(params)
    mses = smoe.get_mses()
    losses = smoe.get_losses()
    num_pis = smoe.get_num_pis()

    cp = {'params': params, 'mses': mses, 'losses': losses, 'num_pis': num_pis}
    with open(path, 'wb') as fd:
        pickle.dump(cp, fd)


def load_params(path):
    with open(path, 'rb') as fd:
        params = pickle.load(fd)['params']
    return params
