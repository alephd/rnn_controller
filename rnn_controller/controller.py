from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple

import tensorflow as tf

from rnn_controller.constants import FLOAT_TYPE
from rnn_controller.response import ResponseState
from rnn_controller.utils import set_tf_tensor, RNNCellInterface, tfnamedtuple, upgrade_cell

EPSILON_VOLUME = 1e-3


def add_summary_variables(cell):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_shape):
            output = func(input_shape)
            [tf.summary.histogram(f"control_weights_{v.name.split('/')[-1]}", v) for v in cell.trainable_variables]
            return output

        return wrapper

    return decorator


class ControllerCell(RNNCellInterface):
    NAME = "controller_cell"
    ControllerState = tfnamedtuple("ControllerSubState", ("realized_volume_cum", "current_volume_target"))

    def __init__(self, volume_target, control, total_time,
                 volume_pattern_provider=None, relative=False,
                 extra_control_input_names=(),
                 trainable=True, name=NAME, dtype=FLOAT_TYPE):
        self._volume_target = volume_target
        self._control = upgrade_cell(control, "control", "control_output", root=name)
        self._control.build = add_summary_variables(self._control)(self._control.build)
        assert self._control.output_size > 1
        self.State = tfnamedtuple("ControllerState", self._control.state_name + self.ControllerState._fields)
        initial_state_params = self.control.initial_state_params + ((0., False), (0., False))
        super(ControllerCell, self).__init__(initial_state_params, state_size=self._get_state_size(),
                                             trainable=trainable, name=name, dtype=dtype)
        self._total_time = total_time
        if volume_pattern_provider is None:
            volume_pattern_provider = lambda current: (self.total_time - current)
        self._volume_pattern_provider = volume_pattern_provider
        self.relative = relative
        self._epsilon_volume = EPSILON_VOLUME
        self._control_input_names = ('remaining_time', 'current_volume_target', 'volume',) + extra_control_input_names
        self.t_0 = 0

    @property
    def trainable_weights(self):
        return self.control.trainable_weights + super().trainable_weights

    @property
    def non_trainable_weights(self):
        return self.control.non_trainable_weights + super().non_trainable_weights

    @property
    def losses(self):
        return self.control.losses + super().losses

    @property
    def control(self):
        return self._control

    @property
    def total_time(self):
        return self._total_time

    @property
    def volume_target(self):
        if self.built is False:
            raise ValueError("Illegal state, cell hasn't been built.")
        return self._volume_target

    @RNNCellInterface.root.setter
    def root(self, root):
        self._root = root
        self.control.root = self.full_name

    def _get_state_size(self):
        control_state_size = self.control.state_size
        if not hasattr(control_state_size, '__len__'):
            control_state_size = (control_state_size,)
        return self.State(*control_state_size, realized_volume_cum=1, current_volume_target=1)

    def get_control_state(self, state):
        return self.control.State(*state[:-len(self.ControllerState._fields)])

    def make_state(self, control_state, volume_cum, current_volume_target):
        return self.State(*control_state, volume_cum, current_volume_target)

    def build(self, inputs_shape):
        super(ControllerCell, self).build(inputs_shape)
        set_tf_tensor(self, '_volume_target', dtype=self.dtype)
        set_tf_tensor(self, '_total_time', dtype=self.dtype)
        self.t_0 = tf.zeros(1, self.dtype)

    def call(self, response: tf.Tensor, state: namedtuple, training=False) -> (tf.Tensor, namedtuple):
        *control_state, realized_volume_cum, _ = state
        volume, cost, t = ResponseState.extract_data(response)
        remaining_time = tf.identity((self.total_time - t) / self.total_time, "remaining_time")
        volume_cum = tf.identity(realized_volume_cum + volume, name="volume_cum")
        # time_of_day_ratio = full_pattern / remaining_pattern_part
        time_of_day_ratio = self._volume_pattern_provider(self.t_0) / self._volume_pattern_provider(t)
        # intraday_adjustment_factor is (unit-less) relative progress
        # compute instantaneous volume target (volume/time sampling) to compare with feedback volume on the sample
        current_volume_target = time_of_day_ratio * (self._volume_target - volume_cum) / self._total_time
        if self.relative:
            cvt = tf.identity(current_volume_target / self.volume_target, name="relative_current_target")
            volume = tf.identity(volume / self.volume_target, name="relative_volume")
            cost = tf.identity(cost / self.volume_target, name="relative_cost")
        else:
            cvt = current_volume_target
        control_input = self.build_control_input(remaining_time=remaining_time, current_volume_target=cvt,
                                                 volume_target=self.volume_target * tf.ones_like(volume,
                                                                                                 dtype=self.dtype),
                                                 volume=volume, cost=cost)
        control_output, control_state = self.control(control_input, self.control.State(*control_state), training)
        return control_output, self.State(*control_state, realized_volume_cum=volume_cum,
                                          current_volume_target=current_volume_target)

    def build_control_input(self, **kwargs):
        return tf.concat([kwargs[k] for k in self._control_input_names], axis=1, name="control_input")


class ControllerRNNCell(RNNCellInterface):
    NAME = "controller_rnn_cell"
    ControllerState = tfnamedtuple("ControllerSubState", ("realized_volume_cum",))

    def __init__(self, volume_target, cell, total_time, relative=False,
                 extra_control_input_names=(),
                 trainable=True, name=NAME, dtype=FLOAT_TYPE, K=3):
        self._volume_target = volume_target
        self._cell = upgrade_cell(cell, "control", "control_output", root=name)
        self.State = tfnamedtuple("ControllerState", self._cell.state_name + self.ControllerState._fields)
        initial_state_params = self.control.initial_state_params + ((0, False),)
        super(ControllerRNNCell, self).__init__(initial_state_params, state_size=self._get_state_size(),
                                                trainable=trainable, name=name, dtype=dtype)
        self._total_time = total_time
        self.relative = relative
        self._epsilon_volume = EPSILON_VOLUME
        self._control_input_names = ('remaining_time', 'remaining_volume',) + extra_control_input_names
        if self.with_time_embedding:
            self.K = K
        else:
            self.K = 1
        self.time_embedding = None

    @property
    def trainable_weights(self):
        return self.control.trainable_weights + super().trainable_weights

    @property
    def non_trainable_weights(self):
        return self.control.non_trainable_weights + super().non_trainable_weights

    @property
    def losses(self):
        return super().losses + self._cell.losses

    @property
    def sub_state_size(self):
        return len(self.ControllerState._fields)

    @property
    def with_time_embedding(self):
        return "time_embedding" in self._control_input_names

    @property
    def control(self):
        return self._cell

    @property
    def total_time(self):
        return self._total_time

    @property
    def volume_target(self):
        if self.built is False:
            raise ValueError("Illegal state, cell hasn't been built.")
        return self._volume_target

    @RNNCellInterface.root.setter
    def root(self, root):
        self._root = root
        self.control.root = self.full_name

    def _get_state_size(self):
        control_state_size = self._cell.state_size
        if not hasattr(control_state_size, '__len__'):
            control_state_size = (control_state_size,)
        return self.State(*control_state_size, realized_volume_cum=1)

    def get_control_state(self, state):
        return self.control.State(*state[:-self.sub_state_size])

    def make_state(self, control_state, remaining_volume):
        return self.State(*control_state, remaining_volume)

    def build(self, inputs_shape):
        super(ControllerRNNCell, self).build(inputs_shape)
        set_tf_tensor(self, '_volume_target', dtype=self.dtype)
        set_tf_tensor(self, '_total_time', dtype=self.dtype)
        if self.with_time_embedding:
            self.time_embedding = self.add_weight(
                "time_embedding", (self.K, 288), initializer=tf.ones_initializer(dtype=self.dtype))  # FIXME(nperrin16)

    def call(self, response: tf.Tensor, state: namedtuple, training=False) -> (tf.Tensor, namedtuple):
        *control_state, realized_volume_cum = state
        volume, cost, time = ResponseState.extract_data(response)
        volume_cum = tf.identity(realized_volume_cum + volume, name="volume_cum")
        remaining_time = tf.identity((self.total_time - time) / self.total_time, "remaining_time")
        remaining_volume = tf.identity(self.volume_target - volume_cum, name="remaining_volume")
        if self.relative:
            remaining_volume = tf.identity(remaining_volume / self.volume_target, name="relative_remaining_volume")
            volume = tf.identity(volume / self.volume_target, name="relative_volume")
            cost = tf.identity(cost / self.volume_target, name="relative_cost")
        if self.time_embedding is None:
            emb = None
        else:
            emb = tf.ones([tf.shape(response)[0], 1], dtype=self.dtype
                          ) * self.time_embedding[:, tf.cast(time[0][0], tf.int32)]
        control_input = self.build_control_input(remaining_time=remaining_time, remaining_volume=remaining_volume,
                                                 volume_target=self.volume_target * tf.ones_like(volume,
                                                                                                 dtype=self.dtype),
                                                 volume=volume,
                                                 cost=cost,
                                                 time_embedding=emb)
        control_output, control_state = self._cell(control_input, self.control.State(*control_state), training)
        return control_output, self.State(*control_state, realized_volume_cum=volume_cum)

    def build_control_input(self, **kwargs):
        return tf.concat([kwargs[k] for k in self._control_input_names], axis=1, name="control_input")


def compute_error(states):
    return ResponseState.extract_volume(states.response) - states.current_volume_target
