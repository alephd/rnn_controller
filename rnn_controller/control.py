from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import tensorflow as tf

from rnn_controller.constants import FLOAT_TYPE, MIN_NOISE_VALUE, TF_RANDOM_SEED
from rnn_controller.utils import set_tf_tensor, LambdaWrapperCell, RNNCellInterface, tfnamedtuple, clip_bid, \
    upgrade_cell, control_output_constraint

ALLOWED_STACKED_RNN_ARGS = {
    'cells',
    'trainable',
    'name',
    'dtype',
    'input_shape',
    'batch_input_shape',
    'batch_size',
    'weights',
    'activity_regularizer',
}


class PIControlLevelCell(RNNCellInterface):

    NAME = "pi_control_level_cell"
    State = tfnamedtuple("PiControlState", ("control_output", "i_error"))

    # TODO(nperrin16): Pay attention to the mutable dict.
    def __init__(self, k_l, t_i, t_s, plant_gain_var=1., plant_gain_var_constraint=None, use_log=False,
                 signal_bias=None, initial_state_params=(
                    (0., False, {'constraint': lambda u_level: clip_bid(u_level)}), (0., False)),
                 trainable=True, name=NAME, dtype=FLOAT_TYPE):
        """
            k_l: large -> fast convergence but less robustness
            t_i: small -> fast convergence but more volatility
        """
        super().__init__(initial_state_params=initial_state_params, state_size=self.State(control_output=1, i_error=1),
                         trainable=trainable, name=name, dtype=dtype)
        self.use_log = use_log
        self.k_l = k_l
        self.t_i = np.log(t_i) if use_log else t_i
        self.t_s = t_s
        self.plant_gain_var = np.log(plant_gain_var) if use_log else plant_gain_var
        self.plant_gain_var_constraint = plant_gain_var_constraint
        if use_log:
            self.plant_gain = lambda u: np.exp(self.plant_gain_var)
            self.k_i = t_s / np.exp(t_i)
        else:
            self.plant_gain = lambda u: self.plant_gain_var
            self.k_i = t_s / t_i
        self.signal_bias = signal_bias

    def build(self, inputs_shape):
        super().build(inputs_shape)
        set_tf_tensor(self, 'k_l', dtype=self.dtype)
        set_tf_tensor(self, 't_s', dtype=self.dtype)
        self.t_i = self.add_weight(
            't_i', shape=(), dtype=self.dtype, trainable=True,
            initializer=tf.constant_initializer(self.t_i, dtype=self.dtype))
        tf.summary.scalar('control/t_i', tf.exp(self.t_i) if self.use_log else self.t_i)
        self.plant_gain_var = self.add_weight(
            'plant_gain', shape=(), dtype=self.dtype, trainable=True,
            initializer=tf.constant_initializer(self.plant_gain_var, dtype=self.dtype),
            constraint=self.plant_gain_var_constraint)
        tf.summary.scalar('control/plant_gain', tf.exp(self.plant_gain_var) if self.use_log else self.plant_gain_var)
        if self.use_log:
            self.plant_gain = lambda _control_signal_level: tf.exp(self.plant_gain_var)
            self.k_i = tf.identity(self.t_s / tf.exp(self.t_i), name='k_i')
        else:
            self.plant_gain = lambda _control_signal_level: self.plant_gain_var
            self.k_i = tf.identity(self.t_s / self.t_i, name='k_i')
        if self.signal_bias is not None:
            self.signal_bias = self.add_weight(
                'signal_bias', shape=(), dtype=self.dtype, trainable=True,
                initializer=tf.constant_initializer(self.signal_bias, dtype=self.dtype))
        else:
            self.signal_bias = 0.
            set_tf_tensor(self, 'signal_bias', dtype=self.dtype)
        tf.summary.scalar('control/signal_bias', self.signal_bias)

    def call(self, control_input: tf.Tensor, state: State, training=False) -> (tf.Tensor, State):
        """State is the integral term `i_error`."""
        control_signal_level, i_error = state
        k_p = self.k_l / self.plant_gain(control_signal_level)
        current_volume_target = tf.identity(control_input[:, 1], name="volume_target")
        volume = tf.identity(control_input[:, 2], name="volume")
        error = tf.identity(current_volume_target - volume, name="error")[:, tf.newaxis]
        p_error = k_p * error
        i_error = i_error + p_error * self.k_i
        control_output = clip_bid(self.signal_bias + p_error + i_error, name='control_signal_level')
        return control_output, self.State(control_output=control_output, i_error=i_error)


class PIDControlLevelCell(RNNCellInterface):

    NAME = "pid_control_level_cell"
    State = tfnamedtuple("PiControlState", ("control_output", "i_error", "previous_error"))

    # TODO(nperrin16): Pay attention to the mutable dict.
    def __init__(self, k_l, t_i, t_s, t_d=1e-16, plant_gain_var=1., plant_gain_var_constraint=None, use_log=False,
                 signal_bias=None, initial_state_params=(
                    (0., False, {'constraint': lambda u_level: clip_bid(u_level)}), (0., False), (0., False)),
                 trainable=True, name=NAME, dtype=FLOAT_TYPE):
        """
            k_l: large -> fast convergence but less robustness
            t_i: small -> fast convergence but more volatility
        """
        super().__init__(initial_state_params=initial_state_params,
                         state_size=self.State(control_output=1, i_error=1, previous_error=1),
                         trainable=trainable, name=name, dtype=dtype)
        self.use_log = use_log
        self.k_l = k_l
        self.t_i = np.log(t_i) if use_log else t_i
        self.t_d = np.log(t_d) if use_log else t_d
        self.t_s = t_s
        self.plant_gain_var = np.log(plant_gain_var) if use_log else plant_gain_var
        self.plant_gain_var_constraint = plant_gain_var_constraint
        if use_log:
            self.plant_gain = lambda u: np.exp(self.plant_gain_var)
            self.k_i = t_s / np.exp(t_i)
            self.k_d = t_s * np.exp(t_d)
        else:
            self.plant_gain = lambda u: self.plant_gain_var
            self.k_i = t_s / t_i
            self.k_d = t_s * t_d
        self.signal_bias = signal_bias

    def build(self, inputs_shape):
        super().build(inputs_shape)
        set_tf_tensor(self, 'k_l', dtype=self.dtype)
        set_tf_tensor(self, 't_s', dtype=self.dtype)
        self.t_i = self.add_weight(
            't_i', shape=(), dtype=self.dtype, trainable=True,
            initializer=tf.constant_initializer(self.t_i, dtype=self.dtype))
        tf.summary.scalar('control/t_i', tf.exp(self.t_i) if self.use_log else self.t_i)
        self.t_d = self.add_weight(
            't_d', shape=(), dtype=self.dtype, trainable=True,
            initializer=tf.constant_initializer(self.t_d, dtype=self.dtype))
        tf.summary.scalar('control/t_d', tf.exp(self.t_d) if self.use_log else self.t_d)
        self.plant_gain_var = self.add_weight(
            'plant_gain', shape=(), dtype=self.dtype, trainable=True,
            initializer=tf.constant_initializer(self.plant_gain_var, dtype=self.dtype),
            constraint=self.plant_gain_var_constraint)
        tf.summary.scalar('control/plant_gain', tf.exp(self.plant_gain_var) if self.use_log else self.plant_gain_var)
        if self.use_log:
            self.plant_gain = lambda _control_signal_level: tf.exp(self.plant_gain_var)
            self.k_i = tf.identity(self.t_s / tf.exp(self.t_i), name='k_i')
            self.k_d = tf.identity(self.t_s * tf.exp(self.t_d), name='k_d')
        else:
            self.plant_gain = lambda _control_signal_level: self.plant_gain_var
            self.k_i = tf.identity(self.t_s / self.t_i, name='k_i')
            self.k_d = tf.identity(self.t_s * self.t_d, name='k_i')
        if self.signal_bias is not None:
            self.signal_bias = self.add_weight(
                'signal_bias', shape=(), dtype=self.dtype, trainable=True,
                initializer=tf.constant_initializer(self.signal_bias, dtype=self.dtype))
        else:
            self.signal_bias = 0.
            set_tf_tensor(self, 'signal_bias', dtype=self.dtype)
        tf.summary.scalar('control/signal_bias', self.signal_bias)

    def call(self, control_input: tf.Tensor, state: State, training=False) -> (tf.Tensor, State):
        """State is the integral term `i_error`."""
        control_signal_level, i_error, previous_error = state
        k_p = self.k_l / self.plant_gain(control_signal_level)
        current_volume_target = tf.identity(control_input[:, 1], name="volume_target")
        volume = tf.identity(control_input[:, 2], name="volume")
        error = tf.identity(current_volume_target - volume, name="error")[:, tf.newaxis]
        p_error = k_p * error
        i_error = i_error + p_error * self.k_i
        d_error = (error - previous_error) * self.k_d
        control_output = clip_bid(self.signal_bias + p_error + i_error + d_error, name='control_signal_level')
        return control_output, self.State(control_output=control_output, i_error=i_error, previous_error=error)


class PIDControlCell(PIDControlLevelCell):

    NAME = "pid_control_cell"

    def __init__(self, *args, name=NAME, initial_state_params=(
            ([0., 0.2], False, {"constraint": lambda u: control_output_constraint(u, np.infty, MIN_NOISE_VALUE, 1.)}),
            (0., False), (0., False)), **kwargs):
        super().__init__(*args, initial_state_params=initial_state_params, name=name, **kwargs)
        self._state_size = self.State(control_output=2, i_error=1, previous_error=1)

    def call(self, control_input: tf.Tensor, state: PIDControlLevelCell.State, training=False)\
            -> (tf.Tensor, PIDControlLevelCell.State):
        """State is the (integral term `i_error`, previous_error)."""
        previous_control_output, i_error, previous_error = state
        control_signal_level = tf.identity(previous_control_output[:, 0:1], name='control_signal_level')
        control_signal_noise = tf.identity(previous_control_output[:, 1:2], name='control_signal_noise')
        control_signal_level, state = super().call(control_input,
                                                   PIDControlLevelCell.State(control_signal_level,
                                                                            i_error,
                                                                            previous_error))
        control_output = tf.concat([control_signal_level, control_signal_noise], axis=1, name='control_output')
        return control_output, state._replace(control_output=control_output)


class PIControlCell(PIControlLevelCell):

    NAME = "pi_control_cell"

    def __init__(self, *args, name=NAME, initial_state_params=(
            ([0., 0.2], False, { "constraint": lambda u: control_output_constraint(u, np.infty, MIN_NOISE_VALUE, 1.)}),
            (0., False)), **kwargs):
        super().__init__(*args, initial_state_params=initial_state_params, name=name, **kwargs)
        self._state_size = self.State(control_output=2, i_error=1)

    def call(self, control_input: tf.Tensor, state: PIControlLevelCell.State, training=False)\
            -> (tf.Tensor, PIControlLevelCell.State):
        """State is the integral term `i_error`."""
        previous_control_output, i_error = state
        control_signal_level = tf.identity(previous_control_output[:, 0:1], name='control_signal_level')
        control_signal_noise = tf.identity(previous_control_output[:, 1:2], name='control_signal_noise')
        control_signal_level, state = super().call(control_input, PIControlLevelCell.State(control_signal_level, i_error))
        control_output = tf.concat([control_signal_level, control_signal_noise], axis=1, name='control_output')
        return control_output, state._replace(control_output=control_output)


class ConstantNoiseCell(RNNCellInterface):

    NAME = "constant_noise_cell"

    def __init__(self, cell,
                 initial_state_params=(0.1, True,
                                       {'constraint': lambda noise: tf.clip_by_value(noise, 0.01, 1.)}),
                 dtype=FLOAT_TYPE,
                 **kwargs):
        self.cell = upgrade_cell(cell, "control", "control_output", root=self.NAME)
        self.State = self.cell.State
        self._initial_state_params = self.State(*self.cell.initial_state_params)
        self._initial_state_params = self._initial_state_params._replace(
            control_output=self._merge_params(self.initial_state_params.control_output,
                                              initial_state_params, idx=0))
        kwargs['name'] = self.NAME
        super().__init__(self.initial_state_params, state_size=self.state_size, dtype=dtype, **kwargs)

    @staticmethod
    def _merge_params(p1, p2, idx=0):
        value1, trainable1, *other1 = p1
        value2, trainable2, *other2 = p2
        constraint1 = other1[0]['constraint'] if other1 else None
        constraint2 = other2[0]['constraint'] if other2 else None
        if not hasattr(value1, '__len__'):
            value1 = [value1]
        if not hasattr(value2, '__len__'):
            value2 = [value2]

        if constraint1 is not None or constraint2 is not None:
            def apply_if_not_none(constraint, x):
                return constraint(x) if constraint is not None else x

            def constraint(u):
                with tf.name_scope("merged_constraint"):
                    constrained1 = apply_if_not_none(constraint1, tf.concat([u[0:1], u[2:]], axis=0))
                    constrained2 = apply_if_not_none(constraint2, u[1:2])
                return tf.concat([constrained1[0:1], constrained2, constrained1[1:]], axis=0,
                                 name="merged_constraint")

            return np.concatenate((value1[idx:idx + 1], value2, value1[:idx], value1[idx + 1:])), \
                   trainable1 and trainable2, {'constraint': constraint}
        return np.concatenate(
            (value1[idx:idx + 1], value2, value1[:idx], value1[idx + 1:])), trainable1 and trainable2

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
    def state_size(self):
        internal_state_size = self.cell.state_size
        if not hasattr(internal_state_size, '__len__'):
            internal_state_size = (internal_state_size,)
        state_size = self.State(*internal_state_size)
        return state_size._replace(control_output=self.cell.output_size + 1)

    def _get_wrapped_state_size(self):
        wrapped_state_size = self.cell.state_size
        if not hasattr(wrapped_state_size, '__len__'):
            wrapped_state_size = (wrapped_state_size,)
        return self.cell.State(*wrapped_state_size)

    def call(self, control_input: tf.Tensor, state: tfnamedtuple, training=False) -> (tf.Tensor, tfnamedtuple):
        cell_output_state, control_signal_noise = self._deconcat(state.control_output)
        cell_state_in = state._replace(control_output=cell_output_state)
        cell_output, cell_states = self.cell(control_input, cell_state_in, training=training)
        control_output = self._concat(cell_output, control_signal_noise)
        return control_output, self.State(*cell_states)._replace(control_output=control_output)

    @staticmethod
    def _deconcat(control_output):
        control_signal_level = tf.identity(control_output[:, 0][:, tf.newaxis], name='control_signal_level')
        control_signal_noise = tf.identity(control_output[:, 1][:, tf.newaxis], name='control_signal_noise')
        cell_output_state = tf.concat([control_signal_level, control_output[:, 2:]], axis=1, name='cell_output_state')
        return cell_output_state, control_signal_noise

    @staticmethod
    def _concat(cell_output, control_signal_noise):
        return tf.concat([cell_output[..., 0:1], control_signal_noise, cell_output[..., 1:]], axis=-1,
                         name='control_output')


def create_control(control_type, params):
    params = params.copy()
    control_output_constraint = params.pop("control_output_constraint")
    if control_type.startswith("constant_noise_"):
        cell_type = control_type[len("constant_noise_"):]
        constant_noise_params = params.pop("constant_noise_params", {})
        params["control_output_constraint"] = control_output_constraint
        return ConstantNoiseCell(create_control(cell_type, params), **constant_noise_params)
    elif control_type.startswith("stacked_"):
        # 'stacked_lstm8_gru2' will create a StackedRNNCell with: [LSTMCell(8), GRUCell(2)]
        cell_types = control_type.split('_')[1:]
        cells = []
        for i, c_t in enumerate(cell_types):
            cell_type, num_unit = re.findall(r'\d+|\D+', c_t)
            cell_params = {"units": int(num_unit), **params}
            # If no initializer is specified, make sure the seed is set to make runs repeatable
            cell_params['kernel_initializer'] = cell_params.get(
                'kernel_initializer', tf.keras.initializers.glorot_uniform(seed=TF_RANDOM_SEED, dtype=FLOAT_TYPE))
            cell_params['recurrent_initializer'] = cell_params.get(
                'recurrent_initializer', tf.keras.initializers.orthogonal(seed=TF_RANDOM_SEED, dtype=FLOAT_TYPE))
            if i == len(cell_types) - 1:
                cell_params["activation"] = params.pop("activation")
            # TODO(nperrin16): add the activity_regularization.
            cells.append(get_cell(cell_type)(**cell_params))
        params = {k: v for k, v in params.items() if k in ALLOWED_STACKED_RNN_ARGS}
        if 'cells' not in params:
            params["cells"] = cells
    control_cell = get_cell(control_type)(**params)
    control_cell = LambdaWrapperCell(control_cell, control_output_constraint, dtype=params["dtype"])
    return control_cell


def get_cell(cell_type):
    if cell_type.startswith("lstm"):
        return tf.keras.layers.LSTMCell
    elif cell_type.startswith("gru"):
        return tf.keras.layers.GRUCell
    elif cell_type.startswith("simple"):
        return tf.keras.layers.SimpleRNNCell
    elif cell_type.startswith("stacked"):
        return tf.keras.layers.StackedRNNCells
    elif cell_type.startswith("pid"):
        return PIDControlCell
    elif cell_type.startswith("pi"):
        return PIControlCell
    else:
        raise ValueError("Unknown cell_type: {}".format(cell_type))
