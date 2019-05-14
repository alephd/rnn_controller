from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rnn_controller.constants import FLOAT_TYPE, INT_TYPE, EPSILON_BID_SCALE
from rnn_controller.utils import EmptyStateCell


class BaseActuatorCell(EmptyStateCell):

    def __init__(self, bid_scale, name, trainable=True, dtype=FLOAT_TYPE):
        self.bid_scale = np.array(bid_scale)
        self.grid_size_bid = self.bid_scale.shape[0]
        super(BaseActuatorCell, self).__init__(trainable=trainable, state_size=(self.bid_scale.shape[0],), name=name,
                                               dtype=dtype)

    def build(self, inputs_shape):
        # Make sure bid_scale > EPSILON_BID_SCALE to avoid numerical instability around 0
        self.bid_scale = tf.maximum(tf.constant(self.bid_scale, name="bid_scale", dtype=self.dtype), EPSILON_BID_SCALE)
        self.grid_size_bid = tf.shape(self.bid_scale)[0]
        super(BaseActuatorCell, self).build(inputs_shape)

    def call(self, control_output: tf.Tensor, state: EmptyStateCell.State, training=False) -> (tf.Tensor, EmptyStateCell.State):
        weight = self.call_implementation(control_output, state, training)
        return weight, state

    def call_implementation(self, control_output: tf.Tensor, state: EmptyStateCell.State, training=False) -> tf.Tensor:
        return NotImplemented


class DiracActuatorWrapperCell(BaseActuatorCell):

    NAME = "dirac_wrapper_"

    def __init__(self, wrapped_actuator, trainable=True, name=None, dtype=FLOAT_TYPE):
        if name is None:
            name = self.NAME + wrapped_actuator.NAME
        super(DiracActuatorWrapperCell, self).__init__(bid_scale=wrapped_actuator.bid_scale, trainable=trainable, name=name, dtype=dtype)
        self.wrapped_actuator = wrapped_actuator

    def build(self, inputs_shape):
        self.wrapped_actuator.build(inputs_shape)
        super(DiracActuatorWrapperCell, self).build(inputs_shape)

    def call_implementation(self, control_output: tf.Tensor, state: BaseActuatorCell.State, training=False) -> tf.Tensor:
        if training:
            weight = self.wrapped_actuator.call_implementation(control_output, state, training)
        else:
            control_signal_level = control_output[:, 0:1]
            index = tf.reduce_sum(tf.cast(tf.less_equal(self.bid_scale, control_signal_level), INT_TYPE), axis=1) - 1
            weight = tf.one_hot(index, self.grid_size_bid, dtype=self.dtype)
        return weight


class ConvolutionActuatorCell(BaseActuatorCell):

    def __init__(self, bid_scale, trainable=True, name="", dtype=FLOAT_TYPE):
        super(ConvolutionActuatorCell, self).__init__(bid_scale=bid_scale, trainable=trainable, name=name, dtype=dtype)
        self.kernel = None

    def build(self, inputs_shape):
        self.kernel = tf.reshape(tf.constant([-1., 1.], dtype=self.dtype), [2, 1, 1], "kernel")
        super(ConvolutionActuatorCell, self).build(inputs_shape)

    def call_implementation(self, control_output: tf.Tensor, state: BaseActuatorCell.State, training=False) -> tf.Tensor:
        control_signal_level = tf.identity(control_output[:, 0:1], name='control_signal_level')
        control_signal_noise = tf.identity(control_output[:, 1:2], name='control_signal_noise')
        cdfs = tf.expand_dims(self.get_distribution(control_signal_level, control_signal_noise).cdf(self.bid_scale),
                              axis=2)
        weights = tf.nn.convolution(cdfs, self.kernel, 'VALID', name='convolution')
        # add remaining weight
        remaining_weight = 1 - tf.reduce_sum(weights, axis=1)
        weights = tf.concat([tf.squeeze(weights, axis=2), remaining_weight], axis=1)
        return weights

    def get_distribution(self, control_signal_level: tf.Tensor,
                         control_signal_noise: tf.Tensor) -> tfp.distributions.Distribution:
        return NotImplemented


class LogNormalActuatorCell(ConvolutionActuatorCell):

    NAME = "log_normal_actuator_cell"

    def __init__(self, bid_scale, trainable=True, name=NAME, dtype=FLOAT_TYPE):
        super(LogNormalActuatorCell, self).__init__(bid_scale=bid_scale, trainable=trainable, name=name, dtype=dtype)

    def build(self, inputs_shape):
        super(LogNormalActuatorCell, self).build(inputs_shape)

    def get_distribution(self, control_signal_level: tf.Tensor,
                         control_signal_noise: tf.Tensor) -> tfp.distributions.Distribution:
        return tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Normal(
                loc=tf.log(control_signal_level),
                scale=control_signal_noise),
            bijector=tfp.bijectors.Exp(),
            name="log_normal_transformed_distribution")


class GammaActuatorCell(ConvolutionActuatorCell):

    NAME = "gamma_actuator_cell"

    def __init__(self, bid_scale, trainable=True, name=NAME, dtype=FLOAT_TYPE):
        super(GammaActuatorCell, self).__init__(bid_scale=bid_scale, trainable=trainable, name=name, dtype=dtype)

    def build(self, inputs_shape):
        super(GammaActuatorCell, self).build(inputs_shape)

    def get_distribution(self, control_signal_level: tf.Tensor,
                         control_signal_noise: tf.Tensor) -> tfp.distributions.Distribution:
        with tf.control_dependencies([tf.assert_less_equal(control_signal_noise,
                                                           tf.ones_like(control_signal_noise, dtype=self.dtype),
                             name='assert_control_signal_noise_lower_than1')]):  # avoid loss of numerical precision
            alpha = 1./(control_signal_noise**2)
        beta = alpha / control_signal_level
        return tfp.distributions.Gamma(concentration=alpha, rate=beta, name='gamma_distribution')


def create_actuator(actuator_type, params):
    if actuator_type.startswith("dirac_wrapper_"):
        wrapped_actuator_cell = create_actuator(actuator_type.replace("dirac_wrapper_", ""), params)
        return DiracActuatorWrapperCell(wrapped_actuator=wrapped_actuator_cell)
    elif actuator_type == "log_normal":
        return LogNormalActuatorCell(bid_scale=params["bid_scale"])
    elif actuator_type == "gamma":
        return GammaActuatorCell(bid_scale=params["bid_scale"])
    else:
        raise ValueError
