from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

from rnn_controller.constants import FLOAT_TYPE
from rnn_controller.response import ResponseState
from rnn_controller.utils import RNNCellInterface, tfnamedtuple


# TODO(nperrin16): find a better name, e.g. ClosedLoopSystemCell
class LoopCell(RNNCellInterface):

    NAME = "closed_loop_system_cell"
    Cells = namedtuple("LoopCells", ("response", "actuator", "controller"))

    def __init__(self, controller_cell, actuator_cell, response_cell, trainable=True, name=NAME, dtype=FLOAT_TYPE):
        self._controller_cell = controller_cell
        self._actuator_cell = actuator_cell
        self._response_cell = response_cell
        self._cells = self.Cells(self._response_cell, self._actuator_cell, self._controller_cell)
        for c in self.cells:
            c.root = name
        self.State = tfnamedtuple("LoopState", ["states"])

        class AllStates(tfnamedtuple("AllStatesBase", self.state_name)):

            @property
            def response_volume(self):
                if not hasattr(self, "_response_volume"):
                    self._response_volume = ResponseState.extract_volume(self.response)
                return self._response_volume

        self.AllStates = AllStates
        super().__init__(None, state_size=self.state_size, trainable=trainable, name=name, dtype=dtype)

    @staticmethod
    def make_state(r_state, a_state, c_state):
        return r_state + a_state + c_state

    @staticmethod
    def get_response_output(state):
        return state[0]

    @property
    def losses(self):
        return super().losses + [l for cell in self._cells for l in cell.losses]

    @property
    def cells(self):
        return self._cells

    @property
    def state_size(self):
        return self.State(states=self.output_size)

    @property
    def output_size(self):
        # To keep the tf.RNNCell compatibility. Do we really want to keep it btw?
        return sum([s for c in self.cells for s in c.state_size])

    @property
    def state_name(self):
        return tuple(n for c in self.cells for n in c.state_name)

    @property
    def ranges(self):
        ranges = []
        start = 0
        for s in [s for c in self.cells for s in c.state_size]:
            ranges.append((start, start + s))
            start = start + s
        return self.AllStates(*ranges)

    @property
    def controller_state_len(self):
        return len(self._controller_cell.state_size)

    @property
    def actuator_state_len(self):
        return len(self._actuator_cell.state_size)

    @property
    def response_state_len(self):
        return len(self._response_cell.state_size)

    def call(self, volume_curve, state, training=False):
        state = self.unstack(state[0])
        previous_response = self.get_response_output(state)
        controller_state = self.get_controller_state(state)
        actuator_state = self.get_actuator_state(state)
        response_state = self.get_response_state(state)
        control_output, c_state = self._controller_cell(previous_response, controller_state, training)

        bid_distribution, a_state = self._actuator_cell(control_output, actuator_state, training)
        response, r_state = self._response_cell(
            tf.stack([volume_curve, bid_distribution], axis=1, name="response_input"),
            response_state,
            training)
        output = tf.concat(self.make_state(r_state, a_state, c_state), axis=1, name="loop_output")
        return output, [output]

    def build_initial_state(self):
        [c.build_initial_state() for c in self.cells]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.State(tf.concat([s for c in self.cells for s in c.get_initial_state(
            inputs=inputs, batch_size=batch_size, dtype=dtype)], axis=1))

    def unstack(self, states):
        # Last dimension is supposed to be the concatenation of the various states.
        with tf.name_scope("unstack_state"):
            return self.AllStates(*[states[..., s: e] for s, e in self.ranges])

    def controller_range(self):
        start = self.response_state_len + self.actuator_state_len
        return start, start + self.controller_state_len

    def get_controller_state(self, state):
        start, end = self.controller_range()
        return self._controller_cell.State(*state[start: end])

    def get_actuator_state(self, state):
        start = self.response_state_len
        return self._actuator_cell.State(*state[start: start + self.actuator_state_len])

    def get_response_state(self, state):
        start = 0
        return self._response_cell.State(*state[start: start + self.response_state_len])

    def get_control_state(self, state):
        return self._controller_cell.get_control_state(self.get_controller_state(state))
