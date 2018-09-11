# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Arithmetic Operations that don't fit into math_ops due to dependencies.

To avoid circular dependencies, some math_ops should go here.  Documentation
callouts, e.g. "@@my_op" should go in math_ops.  To the user, these are just
normal math_ops.

CHANGELOG:
This is an slightly adapted version of the original _exponential_space_einsum function of tensorflow supporting
an arbitrary number of unknown dimensions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def exponential_space_einsum(equation, *inputs):
    """Fallback implementation that supports summing an index over > 2 inputs."""
    if '...' in equation:
        raise ValueError("Subscripts with ellipses are not yet supported.")

    match = re.match('([a-z,]+)(->[a-z]*)?', equation)
    if not match:
        raise ValueError(
            'Indices have incorrect format: %s' % equation
        )

    inputs = list(inputs)
    idx_in = match.group(1).split(',')
    idx_all = set(''.join(idx_in))
    indices = ''.join(sorted(idx_all))

    if match.group(2):
        idx_out = match.group(2)[2:]

    else:
        # infer the output subscripts if not given, assume alphabetical order
        counts = {ax: 0 for ax in indices}
        for axes_ in idx_in:
            for ax in axes_:
                counts[ax] += 1

        idx_out = ''.join(sorted(
            ax for ax in indices
            if counts[ax] == 1
        ))

    if len(idx_in) != len(inputs):
        raise ValueError(
            'Expected %d inputs but got %d' % (len(idx_in), len(inputs))
        )

    missing_idx = set(idx_out).difference(idx_all)
    if missing_idx:
        raise ValueError(
            'Unknown output axes: %s' % missing_idx
        )

    axis_order = {}
    for ax in indices:
        if ax not in idx_out:
            axis_order[ax] = len(axis_order)
    for ax in idx_out:
        axis_order[ax] = len(axis_order)

    # transpose inputs so axes are in order
    for i, (input_, axes_) in enumerate(zip(inputs, idx_in)):
        if input_.get_shape().ndims != len(axes_):
            raise ValueError(
                'Input %d with axes %s has incorrect' \
                ' number of dimensions (expected %d, got %d)' % (
                    i, axes_, len(axes_), input_.get_shape().ndims
                )
            )

        sorted_idx = sorted(axes_, key=axis_order.get)

        if len(set(axes_)) != len(axes_):
            raise ValueError(
                'Subscript not supported: an axis appears more than once: %s' % axes_
            )

        if list(axes_) != sorted_idx:
            permuted = [axes_.find(ax) for ax in sorted_idx]
            inputs[i] = array_ops.transpose(input_, permuted)
            idx_in[i] = sorted_idx

    reduction_idx = []
    shapes = [[dim if dim else -1
               for dim in tensor.get_shape().as_list()]
              for tensor in inputs]

    # new
    to_expands = [[] for _ in range(len(shapes))]

    # validate shapes for broadcasting
    for j, ax in enumerate(sorted(idx_all, key=axis_order.get)):
        dims = []
        for i, idx in enumerate(idx_in):
            if ax not in idx:
                shapes[i].insert(j, 1)
                to_expands[i].append(j)
                # print(shapes[i], to_expand[i])
            else:
                dim = shapes[i][j]
                if isinstance(dim, int) and dim > 1:
                    dims.append(dim)

        if len(set(dims)) > 1:
            raise ValueError(
                'Dimension mismatch on axis: %s' % ax
            )

        if ax not in idx_out:
            reduction_idx.append(j)

    # reshape, multiply
    # for input_, shape, to_exp in zip(inputs, shapes, to_expand):
    #  print(input_.shape, shape, to_exp)
    # expanded_inputs = [array_ops.reshape(input_, shape)
    #                   for input_, shape in zip(inputs, shapes)]
    expanded_inputs = [_expand_if_necessary(input_, to_expand)
                       for input_, to_expand in zip(inputs, to_expands)]

    expanded_output = 1
    for input_ in expanded_inputs:
        expanded_output *= input_

    # contract
    return math_ops.reduce_sum(expanded_output, reduction_idx)


def _expand_if_necessary(input_, to_expand):
    '''Expands input_ sequentially in all dims in to_expand if necessary'''
    for dim in to_expand:
        input_ = array_ops.expand_dims(input_, dim)
    return input_