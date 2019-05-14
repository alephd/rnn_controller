from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf

from rnn_controller.utils import tf_diff_axis_1

Result = namedtuple("Result", ["name", "volume", "cost"])
VolumePatternArgs = namedtuple("VolumePatternArgs", ["c1", "c2", "phi1", "phi2"])

VOL_PATTERN_DEFAULT_ARGS = VolumePatternArgs(
    c1=-0.5324402600103008,
    c2=-0.20104511404078226,
    phi1=1.2401621478434632,
    phi2=0.8074286360105021
)


def get_time_index(volumes, target, time_axis=0):
    """Time index on which the target has been reached."""
    # (batch, time, 1)
    time_idx = np.sum(np.cumsum(volumes, axis=time_axis) < target, axis=time_axis) + 1
    if isinstance(time_idx, np.ndarray):
        time_idx[time_idx > volumes.shape[time_axis]] = - 1  # -1 if it hasn't been reached.
    elif isinstance(int(time_idx), int) and time_idx > volumes.shape[time_axis]:
        time_idx = -1
    return time_idx


def get_mask(volumes, target):
    with tf.name_scope("time_mask"):
        return tf.less(tf.concat([volumes[:, 0:1, :], volumes[:, :-1, :]], axis=1), target, name="time_mask")


def compute_stopped_results(states, target, sequence: bool) -> Result:
    """Compute the real strategy results as iif it has stopped once the target was reached."""
    with tf.name_scope("stopped_results"):
        time_mask = get_mask(states.volume, target)
        zeros = tf.zeros_like(states.volume, dtype=states.response.dtype)

        def compute(x):
            return tf.cumsum(tf.where(time_mask, x, zeros), axis=1)

        stopped_cost_cum = compute(tf_diff_axis_1(states.cost, name="stopped_cost"))
        stopped_volume_cum = compute(states.response_volume)
        if sequence:
            return Result(name="stopped_sequence", volume=stopped_volume_cum, cost=stopped_cost_cum)
        else:
            return Result(name="stopped_final", volume=stopped_volume_cum[:, -1, :], cost=stopped_cost_cum[:, -1, :])


def compute_naive_results(volume_curves, target, bid_scale):
    """As fast as you can."""
    time_idx = get_time_index(volume_curves[:, -1], target)
    mask = np.sum(volume_curves[:time_idx, :] < volume_curves[:time_idx, -1:], axis=1)
    cost = np.cumsum(volume_curves[:time_idx, :][np.arange(len(mask)), mask] * bid_scale[mask])
    volume = np.cumsum(volume_curves[:time_idx, :][np.arange(len(mask)), mask])
    return Result(name="naive", cost=cost, volume=volume)


def compute_volume_curve(bid, volume_curves, target, bid_scale):
    bid_idx = np.sum(bid_scale <= bid, axis=0) - 1
    vols = volume_curves[:, bid_idx]
    time_idx = get_time_index(vols, target, time_axis=0)
    return volume_curves[:time_idx, bid_idx]


def compute_volume(bid, volume_curves, target, bid_scale):
    return np.sum(compute_volume_curve(bid, volume_curves, target, bid_scale))


def minimize(volume_curves, target, bid_scale):
    bid_idx = (np.array(
        [compute_volume(b, volume_curves, target, bid_scale) for b in bid_scale]) >= target).nonzero()[0][0]
    return bid_scale[bid_idx]


def compute_optimal_results(volume_curves, target, bid_scale):
    "Optimal constant bid."
    bid = minimize(volume_curves, target, bid_scale)
    volume = compute_volume_curve(bid, volume_curves, target, bid_scale)
    cost = bid * np.cumsum(volume)
    return Result(name="optimal", cost=cost, volume=np.cumsum(volume))


def get_volume_pattern(target, T, vp_args=VOL_PATTERN_DEFAULT_ARGS):
    time = np.linspace(0, 1, num=T)
    return target / T * (1 + vp_args.c1 * np.sin(2 * np.pi * time + vp_args.phi1)
                         + vp_args.c2 * np.sin(4 * np.pi * time + vp_args.phi2))


def get_time_pattern(target, T):
    return target / T * np.ones((T,))


def compute_pattern_results(volume_curves, pattern, bid_scale, prefix=""):
    bid_idx = np.sum(volume_curves < pattern[:, np.newaxis], axis=1)
    bid_idx[bid_idx == volume_curves.shape[1]] = volume_curves.shape[1] - 1
    mask = np.sum(volume_curves[:, :] < volume_curves[np.arange(len(bid_idx)), bid_idx][:, np.newaxis], axis=1)
    return Result(name=f"{prefix}_pattern",
                  cost=np.cumsum(bid_scale[mask] * pattern),
                  volume=np.cumsum(pattern))


def compute_penalty(total_volume, volume_target, penalty):
    with tf.name_scope("penalty"):
        zero = tf.constant(0., dtype=total_volume.dtype)
        return tf.identity(penalty * tf.maximum(zero, volume_target - total_volume), name="penalty")
