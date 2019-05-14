from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from rnn_controller.constants import FLOAT_TYPE
from rnn_controller.utils import RNNCellInterface, tfnamedtuple, tf_diff_axis_1


class ResponseState(tfnamedtuple("ResponseStateBase", ("response", "volume", "cost", "time"))):
    INITIAL_VALUES = ((0., False), (0., False), (0., False), (0., False))

    @staticmethod
    def extract_volume(response):
        return tf.identity(response[..., 0:1], name="response_volume")

    @staticmethod
    def extract_cost(response):
        return tf.identity(response[..., 1:2], name="response_cost")

    @staticmethod
    def extract_time(response):
        return tf.identity(response[..., 2:3], name="response_time")

    @staticmethod
    def extract_data(response):
        return ResponseState.extract_volume(response), ResponseState.extract_cost(response), \
               ResponseState.extract_time(response)

    @property
    def response_cost(self):
        return self.extract_cost(self.response)

    @property
    def realized_volume(self):
        return self.extract_volume(self.response)


class ResponseGeneratorCell(RNNCellInterface):
    NAME = "response_cell"
    State = ResponseState

    def __init__(self, bid_scale, trainable=True, name=NAME, dtype=FLOAT_TYPE):
        state_size = self.State(response=3, cost=1, volume=1, time=1)
        super(ResponseGeneratorCell, self).__init__(ResponseState.INITIAL_VALUES, state_size=state_size,
                                                    trainable=trainable, name=name, dtype=dtype)
        self.bid_scale = bid_scale

    def build(self, input_shape):
        self.bid_scale = tf.expand_dims(tf.constant(self.bid_scale, name="bid_scale", dtype=self.dtype), axis=0)
        super(ResponseGeneratorCell, self).build(input_shape)

    def call(self, inputs: tf.Tensor, state: State, training=False) -> (tf.Tensor, State):
        """inputs: (volume_curve, bid_distribution)"""
        volume_curve = tf.identity(inputs[:, 0, :], name="volume_curve")
        bid_distribution = tf.identity(inputs[:, 1, :], name="bid_distribution")

        # TODO(nperrin16): add an unittest.
        cost_curve = tf.cumsum(self.bid_scale * tf_diff_axis_1(volume_curve, name="volume_distribution"), axis=1)
        response_cost = tf.reduce_sum(bid_distribution * cost_curve, axis=1, name='response_cost', keepdims=True)
        cost = state.cost + response_cost

        response_volume = tf.reduce_sum(volume_curve * bid_distribution, axis=1, keepdims=True, name='response_volume')
        volume = state.volume + response_volume
        time = state.time + 1
        response = tf.concat([response_volume, response_cost, time], axis=1, name="response")
        return response, self.State(response=response, volume=volume, cost=cost, time=time)
