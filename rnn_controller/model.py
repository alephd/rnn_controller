from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import tensorflow as tf
from astropy.visualization import hist
from scipy.stats import gamma

from rnn_controller import tools
from rnn_controller.constants import FLOAT_TYPE
from rnn_controller.controller import compute_error
from rnn_controller.loop import LoopCell
from rnn_controller.response import ResponseGeneratorCell
from rnn_controller.strategy import compute_penalty, compute_stopped_results
from rnn_controller.utils import sum_losses

LOSS_SUMMARY_KEY = 'loss_summaries'
LR_SUMMARY_KEY = "learning_rate_summaries"
GRAD_SUMMARY_KEY = "grad_summaries"
STOPPED_LOSSES = "stopped_losses"


class ControllerModel(tf.keras.models.Model):

    def __init__(self, controller_cell, actuator_cell, missed_imp_penalty, dtype=FLOAT_TYPE, initial_state=None):
        super().__init__()
        self.controller_cell = controller_cell
        self.a_cell = actuator_cell
        self.r_cell = ResponseGeneratorCell(bid_scale=self.a_cell.bid_scale, dtype=dtype)
        self.rnn_cell = LoopCell(self.controller_cell, self.a_cell, self.r_cell, dtype=dtype, name='')
        self.rnn = tf.keras.layers.RNN(self.rnn_cell, dtype=self.dtype, return_sequences=True)
        self.missed_imp_penalty = missed_imp_penalty
        self.initial_state = initial_state
        self._max_budget = None
        self._loss = None

    @property
    def losses(self):
        # We can't use the default property as it relies only on the sublayers losses where in our case the losses are
        # defined at the model level.
        return self._losses + self.rnn.losses

    @property
    def volume_target(self):
        return self.controller_cell.volume_target

    @property
    def max_budget(self):
        if self._max_budget is None:
            self._max_budget = tf.squeeze(self.volume_target * self.missed_imp_penalty, name="max_budget")
        return self._max_budget

    @property
    def loss(self):
        return self._loss

    def call(self, inputs, training=False):
        output = self.rnn(inputs, initial_state=self.initial_state, training=training)
        with tf.name_scope("extract_results"):
            states = self.rnn_cell.unstack(output)  # (batch_size, timesteps, output_size)
            realized_volume = states.volume[:, -1, :]
            total_cost = states.cost[:, -1, :]
        self.add_model_loss(states)
        return realized_volume, total_cost, states

    def add_model_loss(self, states):
        def summary_scalar(name, scalar_tensor):
            tf.summary.scalar(name, scalar_tensor, collections=[LOSS_SUMMARY_KEY], family="losses")

        def l2_loss(name, tensor, collection=tf.GraphKeys.REGULARIZATION_LOSSES):
            loss = tf.norm(tensor, ord=2, name=name)
            tf.losses.add_loss(loss, loss_collection=collection)
            return loss

        losses = []
        with tf.name_scope("losses"):
            with tf.name_scope("model_losses"):
                with tf.name_scope("cost_loss"):
                    cost_loss = tf.losses.compute_weighted_loss(states.cost[:, -1, :])
                    summary_scalar("cost_loss", cost_loss/self.max_budget)
                    losses.append(cost_loss)

                with tf.name_scope("penalty_loss"):
                    missed_imp_loss = tf.losses.compute_weighted_loss(
                        compute_penalty(states.volume[:, -1, :], self.volume_target, self.missed_imp_penalty))
                    summary_scalar("missed_imp_loss", missed_imp_loss/self.max_budget)
                    losses.append(missed_imp_loss)

                self._loss = tf.identity(cost_loss + missed_imp_loss, name="loss")
                summary_scalar("loss", self._loss / self.max_budget)
                tf.losses.add_loss(self._loss)
                losses.append(self._loss)

            with tf.name_scope("norms"):
                with tf.name_scope("control_signal_norm"):
                    reg_loss = l2_loss("control_signal_norm", states.control_output[:, :, 0])
                    summary_scalar("regularization_loss", reg_loss/self.max_budget)
                    losses.append(reg_loss)

                with tf.name_scope("target_error_norm"):
                    if hasattr(states, "current_volume_target"):
                        target_error_norm = l2_loss("target_error_norm", compute_error(states))
                        summary_scalar("target_error_norm", target_error_norm)
                        losses.append(target_error_norm)
                    else:
                        losses.append(tf.zeros((), dtype=self.dtype, name="target_error_norm"))

            with tf.name_scope("stopped_losses"):
                _, _, stopped_cost = compute_stopped_results(states, self.volume_target, sequence=False)
                with tf.name_scope("stopped_cost_loss"):
                    stopped_cost_loss = tf.losses.compute_weighted_loss(stopped_cost, loss_collection=STOPPED_LOSSES)
                    summary_scalar("stopped_cost_loss", stopped_cost_loss/self.max_budget)

                with tf.name_scope("stopped_loss"):
                    stopped_loss = stopped_cost_loss + missed_imp_loss
                    tf.losses.add_loss(stopped_loss, loss_collection=STOPPED_LOSSES)
                    summary_scalar("stopped_loss", stopped_loss/self.max_budget)

        self.add_loss(losses)

    def get_training_loss(self, regularization, spread_reg):
        _, _, loss, reg_loss, target_error_loss, *rnn_cell_losses = self.losses
        with tf.name_scope("losses"):
            regularizers_loss = sum_losses(rnn_cell_losses, name='regularizers_loss', dtype=self.dtype)
            training_loss = loss + regularization * reg_loss / self.missed_imp_penalty \
                            + spread_reg * target_error_loss / self.missed_imp_penalty \
                            + regularizers_loss
            tf.summary.scalar('training_loss', training_loss / self.max_budget,
                              collections=[LOSS_SUMMARY_KEY], family="losses")
        return training_loss

    def add_plot_snapshot(self, states, image_summary_max_state_size=1):

        def plot_scalar_state(figure, state_name, state_val, volume_target, missed_imp_penalty, nb_lines_max=6):
            figure.set_size_inches(10, 5)
            nb_lines = min(nb_lines_max, state_val.shape[0])
            figure.gca().plot(np.transpose(state_val[:nb_lines, :]))
            figure.gca().set_title(f"{state_name.decode('utf-8')} (v={volume_target}, mip={missed_imp_penalty})")
            figure.tight_layout()

        def plot_control_output(figure, state_name, level_val, noise_val, bid_scale_grid,
                                volume_target, missed_imp_penalty, nb_lines_max=6):
            figure.set_size_inches(10, 5)
            nb_lines = min(nb_lines_max, level_val.shape[0])

            concentration = 1. / (noise_val[:nb_lines, :] ** 2)  # alpha
            beta = concentration / level_val[:nb_lines, :]  # beta
            rv = gamma(a=concentration, scale=1 / beta)
            intervals = rv.interval(alpha=0.95)
            for i in range(nb_lines):
                line, = figure.gca().plot(np.transpose(level_val[i, :]))
                figure.gca().plot(np.transpose(intervals[0][i, :]), color=line.get_color(), linestyle=':')
                figure.gca().plot(np.transpose(intervals[1][i, :]), color=line.get_color(), linestyle=':')
            ylims = figure.gca().get_ylim()
            if bid_scale_grid is not None:
                for bid_level in bid_scale_grid.tolist():
                    if bid_level < ylims[0] or bid_level > ylims[1]:
                        continue
                    figure.gca().axhline(y=bid_level, xmin=0, xmax=1, linestyle='--', color='lightgrey')
            figure.gca().set_title(f"{state_name.decode('utf-8')} (v={volume_target}, mip={missed_imp_penalty})")
            figure.tight_layout()

        def plot_state(figures, state_name, state_val, volume_target, missed_imp_penalty, nb_lines_max=6):
            for e, figure in enumerate(figures):
                plot_scalar_state(figure, state_name, state_val[:, :, e],
                                   volume_target, missed_imp_penalty, nb_lines_max)

        def _plot_scalar_state_distribution(figure, state_name, state_val,
                                                    volume_target, missed_imp_penalty, ref):
            figure.set_size_inches(10, 5)
            ax1 = figure.add_subplot(2, 1, 1)
            if state_val.shape[0] > 3:  # use adaptative number of bins
                try:
                    counts, bins, patches = hist(state_val[:, -1, 0], bins='knuth', ax=ax1)
                except ValueError as err:
                    print("Falling back to regular hist, error was:", err)
                    counts, bins, patches = ax1.hist(state_val[:, -1, 0])
            else:
                counts, bins, patches = ax1.hist(state_val[:, -1, 0])
            ax1.set_xlabel(f"{state_name.decode('utf-8')} (v={volume_target}, mip={missed_imp_penalty})")
            ax2 = figure.add_subplot(2, 1, 2)
            ax2.hist(state_val[:, -1, 0], bins=bins, cumulative=True, histtype='step', density=True)
            if ref is not None:
                ax2.vlines(ref, 0, 1, colors='r', linewidth=2)
            ax1.set_xlabel(f"{state_name.decode('utf-8')} (v={volume_target}, mip={missed_imp_penalty})")
            figure.tight_layout()

        plot_scalar_state_distribution = partial(_plot_scalar_state_distribution, ref=None)

        def _plot_summary(*args, **kwargs):
            args = args + (self.volume_target, self.missed_imp_penalty)
            return tools.matplotlib_summary(*args, **kwargs)

        with tf.name_scope("plot_summarize"):
            for state_name, state_value in states._asdict().items():
                if state_name in ['time', 'realized_volume']:
                    continue
                state_size = state_value.shape[2]
                if state_size > image_summary_max_state_size:
                    continue
                with tf.name_scope(f"summarize_{state_name}"):
                    if state_size == 1:  # scalar state
                        _plot_summary(plot_scalar_state, state_name, tf.squeeze(state_value, axis=2),
                                      name=state_name, family=state_name)
                    else:
                        _plot_summary(plot_state, state_name, state_value, max_outputs=state_size,
                                      name=state_name, family=state_name)
            if hasattr(states, "current_volume_target"):
                with tf.name_scope("summarize_cumulated_error"):
                    cum_error = tf.cumsum(compute_error(states), axis=1)
                    _plot_summary(plot_scalar_state, "cumulated_error", tf.squeeze(cum_error, axis=2),
                                  name="cumulated_error", family='cumulated_error')
            _plot_summary(plot_scalar_state, "control_signal_level", states.control_output[..., 0],
                          name="control_signal_level", family="control_output")
            _plot_summary(plot_scalar_state, "control_signal_noise", states.control_output[..., 1],
                          name="control_signal_noise", family="control_output")
            _plot_summary(plot_control_output, "level_and_noise", states.control_output[..., 0],
                          states.control_output[..., 1], self.a_cell.bid_scale,
                          name="level_and_noise", family="control_output")
            tools.matplotlib_summary(_plot_scalar_state_distribution, "volume",
                                     states.volume, self.volume_target,
                                     self.missed_imp_penalty, self.volume_target,
                                     name="volume", family="histograms")
            _plot_summary(plot_scalar_state_distribution, "cost", states.cost, name="cost", family="histograms")
            cost_with_penalty = states.cost + compute_penalty(
                states.volume, self.volume_target, self.missed_imp_penalty)
            _plot_summary(plot_scalar_state_distribution, "cost_with_penalty", cost_with_penalty,
                          name="cost_with_penalty", family="histograms")

            _, _, stopped_cost = compute_stopped_results(states, self.volume_target, sequence=True)
            _plot_summary(plot_scalar_state_distribution, "stopped_cost", stopped_cost,
                          name="stopped_cost", family="histograms")

            max_budget = self.volume_target * self.missed_imp_penalty

            _plot_summary(plot_scalar_state_distribution, "relative_volume", states.volume/self.volume_target,
                          name="volume", family="normed_histograms")
            _plot_summary(plot_scalar_state_distribution, "cost", states.cost/max_budget,
                          name="cost", family="normed_histograms")
            cost_with_penalty = states.cost + compute_penalty(
                states.volume, self.volume_target, self.missed_imp_penalty)
            _plot_summary(plot_scalar_state_distribution, "cost_with_penalty", cost_with_penalty/max_budget,
                          name="cost_with_penalty", family="normed_histograms")

            _, _, stopped_cost = compute_stopped_results(states, self.volume_target, sequence=True)
            _plot_summary(plot_scalar_state_distribution, "stopped_cost", stopped_cost/max_budget,
                          name="stopped_cost", family="normed_histograms")

    def summarize_learning_rate(self, optimizer):
        def summary_scalar(name, tensor):
            tf.summary.scalar(name, tensor, collections=[LR_SUMMARY_KEY], family="learning_rates")

        def summary_histogram(name, tensor):
            tf.summary.histogram(name, tensor, collections=[LR_SUMMARY_KEY], family="learning_rates")

        if isinstance(optimizer, tf.train.RMSPropOptimizer):
            with tf.name_scope("summarize/rms_learning_rate"):
                lr = tf.cast(optimizer._learning_rate_tensor, dtype=self.dtype)
                epsilon = tf.cast(optimizer._epsilon_tensor, dtype=self.dtype)
                for var in self.trainable_weights:
                    # TODO(nperrin16): Fix me
                    try:
                        # lr * grad / sqrt(ms + epsilon)
                        rms = tf.squeeze(optimizer.get_slot(var, "rms"))
                        if tf.rank(rms) == 0:
                            summarize = summary_scalar
                        else:
                            summarize = summary_histogram
                        summarize("lr_over_rms/" + var.name, lr / tf.sqrt(rms + epsilon))
                    except ValueError as e:
                        print(e)
        elif isinstance(optimizer, tf.train.AdamOptimizer):
            with tf.name_scope("summarize/adam_learning_rate"):
                lr_t = tf.cast(optimizer._lr_t, dtype=self.dtype)
                epsilon = tf.cast(optimizer._epsilon_t, dtype=self.dtype)
                # lr_t * m_t / (\sqrt{v_t} + \epsilon)
                for var in self.trainable_weights:
                    try:
                        m_grad = tf.squeeze(optimizer.get_slot(var, "m"))
                        v_grad = tf.squeeze(optimizer.get_slot(var, "v"))
                        if tf.rank(v_grad) == 0 and tf.rank(m_grad) == 0:
                            summarize = summary_scalar
                        else:
                            summarize = summary_histogram
                        summarize("lr/" + var.name, lr_t / tf.sqrt(v_grad + epsilon))
                        summarize("mean_grad/" + var.name, m_grad)
                    except ValueError as e:
                            print(e)

    def compute_gradients(self, optimizer, loss, summarize_gradients=False, clip_norm=None):

        def summary_scalar(name, tensor):
            tf.summary.scalar(name, tensor, collections=[GRAD_SUMMARY_KEY], family="gradients")

        print("All variables:\n"+"\n".join(map(str, tf.trainable_variables())))
        print("Optimized variables:\n"+"\n".join(map(str, self.trainable_weights)))

        grads_and_vars = optimizer.compute_gradients(loss, var_list=self.trainable_weights)
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))
        gradients, variables = zip(*grads_and_vars)
        if summarize_gradients:
            summary_scalar('global_norm', tf.global_norm(gradients))
            for gradient, var in zip(gradients, variables):
                if gradient is None:
                    continue
                summary_scalar("norm/" + var.name.replace(':', '_'), tf.norm(gradient, name='raw_gradient_norm'))
        if clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_norm, name="gradient_clipping")
        if summarize_gradients:
            with tf.name_scope("summarize/gradients"):
                print("Variables and gradients:")
                for gradient, var in zip(gradients, variables):
                    print("\t", var, gradient)
                    if gradient is None:
                        continue
                    s_grad = tf.squeeze(gradient)
                    if tf.rank(s_grad) == 0:
                        summary_scalar(var.name, s_grad)
                    summary_scalar("norm_clipped/"+var.name.replace(':', '_'), tf.norm(gradient, name='gradient_norm'))
        return gradients, variables
