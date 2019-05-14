from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
from tensorflow.python.keras.layers import StackedRNNCells

from rnn_controller.constants import FLOAT_TYPE, MIN_BID_VALUE


def set_tf_tensor(self, name, dtype=FLOAT_TYPE):
    """Convert an attribute to `tf.Tensor`."""
    attr = getattr(self, name)
    if isinstance(attr, tf.Tensor) and attr.dtype != dtype:
        attr = tf.cast(attr, dtype)
    setattr(self, name, tf.convert_to_tensor(attr, name=name, dtype=dtype))


def with_test_scope(test_func):
    """Put the unittest into its own scope to isolate the graph from the other unittests."""

    @functools.wraps(test_func)
    def wrapper(self):
        with tf.variable_scope(test_func.__name__, dtype=FLOAT_TYPE):
            return test_func(self)
    return wrapper


def get_regularization_loss(scope=None, name="total_regularization_loss", dtype=FLOAT_TYPE):
    """Gets the total regularization loss.
    Same as tf.losses.get_regularization_loss with dtype arg.
    Args:
    scope: An optional scope name for filtering the losses to return.
    name: The name of the returned tensor.

    Returns:
    A scalar regularization loss.
    """
    losses = tf.losses.get_regularization_losses(scope)
    if losses:
        return tf.add_n(losses, name=name)
    else:
        return tf.constant(0.0, dtype=dtype, name=name)


def sum_losses(losses, name="total_regularization_loss", dtype=FLOAT_TYPE):
    if losses:
        return tf.add_n(losses, name=name)
    else:
        return tf.constant(0.0, dtype=dtype, name=name)


def clip_bid(x, max_bid=np.infty, name=None):
    return tf.clip_by_value(x, MIN_BID_VALUE, max_bid, name=name)


def control_output_constraint(control_output, max_bid, min_noise, max_noise):
    with tf.name_scope("control_output_constraint"):
        control_signal_level = tf.identity(control_output[:, 0:1], name='control_signal_level')
        control_signal_noise = tf.identity(control_output[:, 1:2], name='control_signal_noise')
        return tf.concat([clip_bid(control_signal_level, max_bid, name="bid_clipping"),
                          tf.clip_by_value(control_signal_noise, min_noise, max_noise, name="noise_clipping"),
                          control_output[:, 2:]], axis=1, name='control_output_constrained')


def smooth_relu(x):
    return tf.where(tf.greater(x, 0), 1 + x, tf.exp(x), name="smooth_relu")


def tfnamedtuple(typename, *field_names, defaults=tuple(), **kwargs):
    """A tfnamedtuple is a namedtuple where tensors if any are wrapped into a `tf.identity` call with the field name."""
    ntuple = namedtuple(typename, *field_names, **kwargs)
    ntuple.__new__.__defaults__ = defaults

    def tensorflowrize(func):
        def wrapper(*args, **kwargs):
            cls, *fargs = args
            new_args = []
            for a, n in zip(fargs, ntuple._fields):
                if isinstance(a, tf.Tensor) and n not in a.name:
                    a = tf.identity(a, name=n)
                new_args.append(a)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, tf.Tensor) and k not in v.name:
                    v = tf.identity(v, name=k)
                new_kwargs[k] = v
            return func(cls, *new_args, **new_kwargs)
        return wrapper
    ntuple.__new__ = tensorflowrize(ntuple.__new__)
    return ntuple


class LambdaWrapperCell(tf.keras.layers.Lambda):
    """Wraps a cell and apply a lambda layer to its output."""

    NAME = 'lambda_cell'

    def __init__(self, cell, *args, **kwargs):
        kwargs['name'] = self.NAME
        self.cell = cell
        self.cell.root = self.NAME
        super().__init__(*args, **kwargs)
        self.root = ''
        self.cell.add_weight = add_scope_to_instance(self.full_name, self.cell.name)(self.cell.add_weight)

    def __getattr__(self, item):
        return getattr(self.cell, item)

    @property
    def trainable_weights(self):
        return super().trainable_weights + self.cell.trainable_weights

    @property
    def non_trainable_weights(self):
        return super().non_trainable_weights + self.cell.non_trainable_weights

    @property
    def losses(self):
        return super().losses + self.cell.losses

    @property
    def full_name(self):
        return os.path.normpath(os.path.join(self.root, self.name))

    def call(self, inputs, *args, **kwargs):
        output, states = self.cell(inputs, *args, **kwargs)
        # TODO(nperrin16) forward potential `mask` argument ?
        output = super().call(output)
        output_idx = get_output_index(self.cell)
        states = tuple([*states[:output_idx], output, *states[output_idx+1:]])
        return output, states


def remove_namescope_overlap(*name_scopes: str) -> str:
    # TODO(nperrin16): improve algo (e.g. using common substring search)
    print("current namescope: ", tf.get_default_graph().get_name_scope())
    print("variable name: ", os.path.join(*name_scopes))
    base_scope = os.path.basename(tf.get_default_graph().get_name_scope())
    name_scopes = list(name_scopes)
    name_scopes_to_keep = [name_scopes.pop()]
    while len(name_scopes) > 0:
        n = name_scopes.pop()
        if n == base_scope or n == '':
            break
        else:
            name_scopes_to_keep.append(n)
    print("returned variable name: ", os.path.normpath(os.path.join(*name_scopes_to_keep[::-1])))
    return os.path.normpath(os.path.join(*name_scopes_to_keep[::-1]))


def _get_name(name, root, instance_name):
    """Remove any scope duplicates in join(name_scope, root/instance_name) and name."""
    names = []
    if not name.startswith(os.path.normpath(os.path.join(root, instance_name))):
        names = [root, instance_name]
    names.extend(name.split("/"))
    return remove_namescope_overlap(*names)


def add_scope_to_instance(root, instance_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(name, *args, **kwargs):
            return func(*args, name=_get_name(name, root, instance_name), **kwargs)
        return wrapper
    return decorator


class RNNCellInterface(tf.keras.layers.Layer):
    """Provides :
        - a default implementation of the `state_name` and `ranges`properties,
    """

    def __init__(self, initial_state_params, state_size, **kwargs):
        super().__init__(**kwargs)
        self._initial_states = self.State(*[None]*len(self.State._fields))
        self._initial_state_params = initial_state_params
        self._state_size = state_size
        self._state_name = self.State._fields
        self._root = ''

    @property
    def initial_state_params(self):
        return self.State(*self._initial_state_params)

    @property
    def initial_states(self):
        return self.State(*self._initial_states)

    @property
    def state_size(self):
        return self.State(*self._state_size)

    @property
    def output_size(self):
        return self.state_size[0]

    @property
    def state_name(self):
        return self._state_name

    @property
    def ranges(self):
        ranges = []
        start = 0
        for s in [s for s in self.state_size]:
            ranges.append((start, start + s))
            start = start + s
        return self.State(*ranges)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def full_name(self):
        return os.path.normpath(os.path.join(self.root, self.name))

    def add_weight(self, name, shape, dtype=None, initializer=None, regularizer=None, trainable=None, constraint=None,
                   **kwargs):
        name = _get_name(name, self.root, self.name)
        return super().add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer,
                                  trainable=trainable, constraint=constraint, **kwargs)

    def build_initial_state(self):
        tmp_initial_state = []
        for s, name, (value, trainable, *params) in zip(self._initial_states, self._state_name, self._initial_state_params):
            if s is None:
                tmp_initial_state.append(self.build_initial(name, params, trainable, value))
        self._initial_states = tuple(tmp_initial_state)

    def build_initial(self, name, other_params, trainable, value):
        print("Build initial state", name, "in", os.path.join(self.root, self.name), ":", value, trainable)
        if other_params:
            other_params = dict(other_params[0])
        else:
            other_params = {}
        _name = other_params.get("name", name)
        other_params["name"] = os.path.join("initial", _name)
        other_params["trainable"] = trainable
        other_params["dtype"] = self.dtype
        other_params["shape"] = (getattr(self._state_size, name, other_params.pop("shape", None)),)
        initial_state = self.add_weight(initializer=tf.constant_initializer(value, dtype=self.dtype), **other_params)
        if trainable:
            for i in range(initial_state.shape[0]):
                tf.summary.scalar(str(i), initial_state[i], family=name)
        return initial_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.State(*self._get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype))

    def _get_initial_state(self, inputs, batch_size, dtype):
        self.build_initial_state()
        batch_size, dtype = self._extract_batch_and_dtype(inputs, batch_size, dtype)
        return [tf.tile(s[tf.newaxis, :], [batch_size, 1]) for s in self._initial_states]

    def zero_state(self, batch_size):
        return self.State(*[tf.zeros((batch_size, s), dtype=self.dtype) for s in self.state_size])

    def _extract_batch_and_dtype(self, inputs, batch_size, dtype):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            if inputs is None:  # We should never be in this case.
                dtype = self.dtype
            else:
                dtype = inputs.dtype
        return batch_size, dtype


class EmptyStateCell(RNNCellInterface):

    State = tfnamedtuple("EmptyState", [])

    def __init__(self, **kwargs):
        super().__init__(tuple(), **kwargs)

    @property
    def state_size(self):
        return self.State()


def upgrade_cell(cell, prefix='', output_name="output", root=''):
    cell = add_state(cell, prefix, output_name)
    cell = add_output_size(cell)
    cell.root = root
    print("upgrading cell: ", cell.name, root)
    cell.add_weight = add_scope_to_instance(root, cell.name)(cell.add_weight)
    return cell


def add_state(cell, prefix, output_name):
    if not hasattr(cell, "State"):
        if hasattr(cell.state_size, '__len__'):
            state_size_len = len(cell.state_size)
        else:
            state_size_len = 1
        new_state_names = tuple(["{prefix}_state_{i}".format(prefix=prefix, i=i) for i in range(0, state_size_len)])
        output_idx = get_output_index(cell)
        new_state_names = tuple([*new_state_names[:output_idx], output_name, *new_state_names[output_idx+1:]])
        cell.State = tfnamedtuple("{}State".format(prefix.title()), new_state_names)
        cell.call = statify_output_decorator(cell, cell.call)
    if not hasattr(cell, 'state_name'):
        cell.state_name = cell.State._fields
    if not hasattr(cell, 'initial_state_params'):
        cell_state_size = cell.state_size
        if not hasattr(cell_state_size, "__len__"):
            cell_state_size = (cell_state_size,)
        #cell_state_size = np.array(cell_state_size).flatten()
        cell.initial_state_params = cell.State(*list((np.zeros((s,)), True) for s in cell_state_size))
    return cell


def add_output_size(cell):
    if not hasattr(cell, 'output_size'):
        if hasattr(cell.state_size, '__len__'):
            cell.output_size = cell.state_size[get_output_index(cell)]
        else:
            cell.output_size = cell.units
    return cell


def get_output_index(cell):
    if isinstance(cell, LambdaWrapperCell):
        return get_output_index(cell.cell)
    if isinstance(cell, StackedRNNCells) and not getattr(cell, "reverse_state_order", True):
        if hasattr(cell.cells[-1], "__len__"):
            num_states = len(cell.cells[-1].state_size)
        else:
            num_states = 1
        return len(cell.state_size) - num_states
    else:
        return 0


def statify_output_decorator(self, call_func):
    def call(inputs, states, training=None, *args, **kwargs):
        output, states = call_func(inputs, self.State(*states), training, *args, **kwargs)
        return output, self.State(*states)
    return call


def rnn_call_decorator(call_func):
    sig = inspect.signature(call_func)
    num_args = len(sig.parameters)

    def call(inputs, states, training=None):
        args = [inputs, states, training]
        output = call_func(*args[:num_args])
        return output, states
    return call


def upgrade_layer_to_cell(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    if not hasattr(layer, "state_size"):
        layer.state_size = 0
        layer.call = rnn_call_decorator(layer.call)
    if not hasattr(layer, "output_size"):
        layer.output_size = layer.units
    return layer


def tf_diff_axis_0(x, name="tf_diff_axis_0"):
    with tf.name_scope(name):
        y = x[1:, ...] - x[:-1, ...]
        return tf.concat([x[0:1, ...], y], axis=0, name=name)


def tf_diff_axis_1(x, name="tf_diff_axis_1"):
    # kernel = tf.reshape(tf.constant([-1., 1.], dtype=x.dtype), [2, 1, 1], "kernel")
    # return tf.nn.convolution(x, kernel, 'SAME', name='convolution')
    with tf.name_scope(name):
        y = x[:, 1:, ...] - x[:, :-1, ...]
        return tf.concat([x[:, 0:1, ...], y], axis=1, name=name)


def tf_diff_axis_2(x, name="diff_axis_2"):
    with tf.name_scope(name):
        y = x[:, :, 1:, ...] - x[:, :, :-1, ...]
        return tf.concat([x[:, :, 0:1, ...], y], axis=2, name=name)


def parse_eval_events(event_path):
    result = {}
    try:
        if event_path is not None:
            for event in tf.train.summary_iterator(event_path):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        result[value.tag] = value.simple_value
    except DataLossError:
        result = {}
    return result
