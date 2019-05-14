from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rnn_controller import utils
from rnn_controller.constants import FLOAT_TYPE, BID_SCALE_EXP, TF_RANDOM_SEED


class ToyVolumeCurvesBuilder(object):

    def __init__(self, initial_volume_curve, func=None):
        self.initial_volume_curve = initial_volume_curve
        self.func = func
        if self.func is None:
            self.func = lambda x, i: x

    def __call__(self, num_time_steps, normalize=None, *args, **kwargs):
        assert(len(self.initial_volume_curve.shape) in [1, 2])
        if len(self.initial_volume_curve.shape) == 1:
            volume_curves = [self.func(self.initial_volume_curve, i) for i in range(0, num_time_steps)]
        else:
            assert self.initial_volume_curve.shape[0] == num_time_steps,\
                f"Num time steps mismatch {self.initial_volume_curve.shape[0]} vs {num_time_steps}"
            volume_curves = [self.func(self.initial_volume_curve[i, :], i) for i in range(0, num_time_steps)]
        factor = 1.
        if normalize is not None:
            daily_volume = np.sum([vc[-1] for vc in volume_curves])
            factor = normalize/daily_volume
        return factor * np.stack(volume_curves, axis=0)


def curve_builder(method):
    """Build a ToyVolumeCurvesBuilder based on the func returned by method."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        shift = method(self, *args, **kwargs)
        builder = ToyVolumeCurvesBuilder(self.base_volume_curve, func=shift)
        return builder(self.num_time_steps, normalize=self.daily_volume)
    return wrapper


def multiplier(method):
    """Preprocess change_times and multipliers arguments and apply curve_builder to the output returned by method. """

    @functools.wraps(method)
    def wrapper(self, change_times, multipliers):
        assert len(change_times) == len(multipliers)
        change_times = np.array(change_times)
        multipliers = np.concatenate(([1.], np.array(multipliers)))
        return curve_builder(method)(self, change_times, multipliers)
    return wrapper


class ToyVolumeCurves(object):
    BASE = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     9., 24., 36., 42., 47., 57., 65., 66., 74., 78., 83., 85., 87., 93., 97., 100., 100., 100., 100.,
                     100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
                     100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
                     100.])

    def __init__(self, daily_volume, num_time_steps, base_volume_curve=BASE, bid_scale=BID_SCALE_EXP):
        self.daily_volume = daily_volume
        self.num_time_steps = num_time_steps
        self.bid_scale = bid_scale
        self.base_volume_curve = base_volume_curve

    def constant_volume(self):
        constant_builder = ToyVolumeCurvesBuilder(self.base_volume_curve)
        return constant_builder(self.num_time_steps)

    @multiplier
    def volume_shocks(self, change_times, multipliers):

        def volume_shift(v, t):
            idx = np.searchsorted(change_times, t, side='right')
            return v * multipliers[idx]
        return volume_shift

    @multiplier
    def shifted_bid_levels(self, change_times, multipliers):

        def bid_shift(v, t):
            idx = np.searchsorted(change_times, t, side='right')
            bid_multiplier = multipliers[idx]
            return bid_multiplication(self.bid_scale, v, bid_multiplier)
        return bid_shift

    @curve_builder
    def volume_pattern(self, volume_pattern_params):
        def volume_pattern(v, t):
            return pattern_evolution(v, t, nb_timestamp=self.num_time_steps, **volume_pattern_params)
        return volume_pattern

    def volume_pattern_shift(self, volume_pattern_params, phase_shift_percent):
        volume_pattern_params['phi1'] += 2 * np.pi * phase_shift_percent
        volume_pattern_params['phi2'] += 4 * np.pi * phase_shift_percent
        return self.volume_pattern(volume_pattern_params)


def pattern_evolution(x, i, c1, phi1, c2, phi2, nb_timestamp):
    pattern = 1. + c1 * np.sin(2 * np.pi * i/nb_timestamp + phi1) + c2 * np.sin(4 * np.pi * i/nb_timestamp + phi2)
    return x * pattern / nb_timestamp


def bid_multiplication(bid_scale, volume_curve, bid_multiplier):
    m_scale = bid_scale * bid_multiplier
    idx = np.sum(bid_scale[:, np.newaxis] <= m_scale, axis=0) - 1
    vc = np.concatenate(([0], volume_curve))
    volumes = vc[1:] - vc[:-1]
    return np.cumsum(np.bincount(idx, weights=volumes, minlength=len(bid_scale)))


class ToyDataset(object):

    SinPatternParams = namedtuple("SinPatternParams", ["c1", "c2", "phi1", "phi2"])

    def __init__(self, count, sample_size, response_type, volume_pattern_type, nb_timestamps, bid_scale,
                 seed=TF_RANDOM_SEED, dtype=FLOAT_TYPE):
        self.sample_size = sample_size
        self.count = count
        self.nb_timestamps = nb_timestamps
        self.bid_scale = bid_scale
        self.response_type = response_type
        self.volume_pattern_type = volume_pattern_type
        self.dtype = dtype
        self.seed = seed
        time = np.linspace(0, 1, num=self.nb_timestamps)[np.newaxis, :]
        self.time = tf.constant(time, shape=(1, self.nb_timestamps), dtype=self.dtype)

    def __call__(self, daily_volume=1000., daily_volume_sigma=0.1, response_params={}, daily_volume_is_random=False,
                 time_growing_uncertainty=None):
        return tf.data.Dataset.range(1).repeat(count=self.count)\
            .flat_map(
            lambda x: self._generate_sample_dataset(daily_volume=daily_volume, daily_volume_sigma=daily_volume_sigma,
                                                    response_params=response_params,
                                                    daily_volume_is_random=daily_volume_is_random,
                                                    time_growing_uncertainty=time_growing_uncertainty))

    def _generate_sample_dataset(self, daily_volume=1000., daily_volume_sigma=0.1, response_params={},
                                 daily_volume_is_random=False, time_growing_uncertainty=None):
        if self.response_type == "grand_canyon":
            response = self.grand_canyon_response(**response_params)
        elif self.response_type == "venice_beach":
            response = self.venice_beach_response(**response_params)
        elif self.response_type == "smooth":
            response = self.smooth_response(**response_params)
        elif self.response_type == "test":
            response = self.test_response(**response_params)
        else:
            raise ValueError('Unknown response_type')
        if self.volume_pattern_type == "usual":
            time_pattern = self.usual_time_pattern()
        elif self.volume_pattern_type == "dromedary":
            time_pattern = self.dromedary_time_pattern()
        elif self.volume_pattern_type == "camel":
            time_pattern = self.camel_time_pattern()
        elif self.volume_pattern_type == "test":
            time_pattern = self.test_time_pattern()
        else:
            raise ValueError('Unknown volume_pattern_type')
        print(time_pattern)
        print(response)

        volume_curves = time_pattern * response
        total_volumes = tf.reduce_sum(time_pattern[:, :, -1:], axis=1, keepdims=True)
        if daily_volume_is_random:
            lognormal_distribution = tfp.distributions.LogNormal(loc=np.log(daily_volume), scale=daily_volume_sigma)
            random_daily_volumes = lognormal_distribution.sample(sample_shape=(self.sample_size, 1, 1), seed=self.seed,
                                                                 name="random_daily_volumes")
            # normal_distribution = tfp.distributions.Normal(loc=0., scale=daily_volume_sigma)
            # random_daily_volumes = daily_volume * tf.exp(
            #     normal_distribution.sample(sample_shape=(self.sample_size, 1, 1), seed=seed, name="random_daily_volumes"))
        else:
            random_daily_volumes = daily_volume * tf.ones((self.sample_size, 1, 1), dtype=self.dtype)

        if time_growing_uncertainty is not None:
            mu = tf.cast(0., dtype=self.dtype)
            sigma = tf.cast(time_growing_uncertainty, dtype=self.dtype)
            dt = tf.cast(1., dtype=self.dtype)
            s_0 = tf.cast(1., dtype=self.dtype)
            normal_distribution = tfp.distributions.Normal(loc=tf.cast(0., dtype=self.dtype), scale=tf.sqrt(dt))
            steps = tf.exp((mu - sigma**2/2) * dt) * tf.exp(
                sigma * normal_distribution.sample(sample_shape=(self.sample_size, self.nb_timestamps - 1, 1),
                                                   seed=self.seed, name='wiener_step'))
            s_t = s_0 * tf.cumprod(tf.concat([tf.ones(shape=(self.sample_size, 1, 1), dtype=self.dtype),
                                              steps], axis=1), axis=1)
            volume_curves = volume_curves * tf.cast(s_t, dtype=self.dtype)

        random_daily_volumes = tf.cast(random_daily_volumes, dtype=self.dtype)
        return tf.data.Dataset.from_tensor_slices(volume_curves/total_volumes * random_daily_volumes)

    def _compute_scale_and_saturation(self, normal_scale, saturation_point):
        lognormal_distribution = tfp.distributions.LogNormal(loc=tf.constant(np.log(saturation_point),
                                                                             dtype=self.dtype),
                                                             scale=normal_scale)
        saturation_points = lognormal_distribution.sample(sample_shape=(self.sample_size, 1), seed=self.seed,
                                                          name="saturation_points")
        saturation_points = tf.tile(saturation_points, multiples=(1, len(self.bid_scale)))
        bid_scales = tf.cast(tf.tile(self.bid_scale[np.newaxis, :], multiples=(self.sample_size, 1)), dtype=self.dtype)
        return bid_scales, saturation_points

    def venice_beach_response(self, saturation_point=5., normal_scale=0.2):
        bid_scales, saturation_points = self._compute_scale_and_saturation(normal_scale, saturation_point)
        bid_scales = bid_scales / saturation_points
        plateau = tf.ones_like(saturation_points, dtype=self.dtype)
        return tf.expand_dims(tf.where(bid_scales >= plateau, plateau, bid_scales),
                              axis=1, name="venice_beach_response")

    def grand_canyon_response(self, saturation_point=5., normal_scale=0.2):
        bid_scales, saturation_points = self._compute_scale_and_saturation(normal_scale, saturation_point)
        response = tf.where(bid_scales >= saturation_points,
                            bid_scales,
                            tf.zeros((self.sample_size, len(self.bid_scale)), dtype=self.dtype))
        response = response / saturation_points * 2.  # renormalize at saturation point
        response = tf.expand_dims(response, axis=1, name="grand_canyon_response")
        return response

    def smooth_response(self, saturation_point=5., normal_scale=0.2):
        bid_scales = tf.cast(tf.tile(self.bid_scale[np.newaxis, :], multiples=(self.sample_size, 1)), dtype=self.dtype)
        bid_scales = bid_scales / saturation_point
        # TODO introduce noise in the slope ?
        return tf.expand_dims(bid_scales, axis=1, name="linear_response")

    def camel_time_pattern(self):
        uniform_phase = tfp.distributions.Uniform(low=tf.constant(0., dtype=self.dtype),
                                                  high=tf.constant(2 * np.pi, dtype=self.dtype))
        uniform_amplitude = tfp.distributions.Uniform(low=tf.constant(-1., dtype=self.dtype),
                                                      high=tf.constant(1., dtype=self.dtype))
        phases = uniform_phase.sample(sample_shape=(self.sample_size, 2), seed=self.seed)
        amplitudes = uniform_amplitude.sample(sample_shape=(self.sample_size, 2), seed=self.seed)
        return tf.expand_dims(self.volume_sin_time_pattern(phi1=phases[:, :1], c1=amplitudes[:, :1],
                                                           phi2=phases[:, 1:], c2=amplitudes[:, 1:]),
                              axis=-1, name="camel_pattern")

    def dromedary_time_pattern(self):
        # exp(1.1 * sin(3.78 + (2 * PI * x) / (60 * 60 * 24)))
        uniform_phase = tfp.distributions.Uniform(low=tf.constant(0., dtype=self.dtype),
                                                  high=tf.constant(2 * np.pi, dtype=self.dtype))
        uniform_amplitude = tfp.distributions.Uniform(low=tf.constant(-2., dtype=self.dtype),
                                                      high=tf.constant(2., dtype=self.dtype))
        phases = uniform_phase.sample(sample_shape=(self.sample_size, 1), seed=self.seed)
        amplitudes = uniform_amplitude.sample(sample_shape=(self.sample_size, 1), seed=self.seed)
        return tf.expand_dims(self.volume_exp_sin_time_pattern(phi1=phases, c1=amplitudes),
                              axis=-1, name="dromedary_pattern")

    def usual_time_pattern(self, pattern_params=SinPatternParams(**{'c1': -0.5324402600103008,
                                                                    'c2': -0.20104511404078226,
                                                                    'phi1': 1.2401621478434632,
                                                                    'phi2': 0.8074286360105021})):
        pattern = self.volume_sin_time_pattern(**pattern_params._asdict())
        return tf.expand_dims(tf.tile(pattern, multiples=(self.sample_size, 1)), axis=-1, name="usual_time_pattern")

    def volume_sin_time_pattern(self, c1, phi1, c2, phi2):
        pattern = 1. + c1 * tf.sin(2 * np.pi * self.time + phi1, name='sin_pattern1') \
               + c2 * tf.sin(4 * np.pi * self.time + phi2, name='sin_pattern2')
        return pattern / self.nb_timestamps

    def volume_exp_sin_time_pattern(self, c1, phi1):
        pattern = tf.exp(c1 * tf.sin(2 * np.pi * self.time + phi1), name='exp_sin_pattern')
        return pattern / self.nb_timestamps

    def test_response(self):
        return tf.cumsum(tf.ones(shape=(self.sample_size, 1, len(self.bid_scale)), dtype=self.dtype),
                         axis=-1, name='response')

    def test_time_pattern(self):
        return tf.ones(shape=(self.sample_size, self.nb_timestamps, 1), dtype=self.dtype, name='time_pattern')

    @staticmethod
    def mixture(dataset1, dataset2):
        return tf.data.experimental.sample_from_datasets((dataset1, dataset2), seed=TF_RANDOM_SEED)


class DataAugmentation(object):

    def __init__(self, shock_probs, nb_timesteps, bid_scale, shock_mult_loc=[0., 0.], shock_mult_scale=[1., 1.],
                 seed=TF_RANDOM_SEED, dtype=FLOAT_TYPE, debug_print=False):
        self.shock_events = tfp.distributions.Bernoulli(probs=tf.cast(shock_probs, dtype))
        self.shock_times = tfp.distributions.Uniform(low=tf.cast([0, 0], dtype),
                                                     high=tf.cast([nb_timesteps-1, nb_timesteps-1], dtype))
        self.shock_multipliers = tfp.distributions.LogNormal(loc=tf.cast(shock_mult_loc, dtype),
                                                             scale=tf.cast(shock_mult_scale, dtype))
        self.seed = seed
        self.dtype = dtype
        self.time = tf.range(0, nb_timesteps, dtype=self.dtype)
        self.ones = tf.ones_like(self.time, dtype=self.dtype)
        self.bid_scale = np.array(bid_scale)
        self._v_idx = 0
        self._b_idx = 1
        self.debug_print = debug_print

    def __call__(self, volume_curve_dataset):
        return self.random_shocks(volume_curve_dataset)

    def random_shocks(self, volume_curve_dataset):
        return volume_curve_dataset.map(map_func=self._random_shocks)

    def _random_shocks(self, volume_curves):
        batch_size = tf.shape(volume_curves)[0]
        shock_event = self.shock_events.sample(sample_shape=(batch_size,), seed=self.seed, name='shock_event')
        shock_time = self.shock_times.sample(sample_shape=(batch_size,), seed=self.seed, name='shock_times')
        shock_multiplier = self.shock_multipliers.sample(sample_shape=(batch_size,), seed=self.seed,
                                                         name='shock_multipliers')
        ones = tf.ones_like(shock_multiplier, dtype=self.dtype)
        if self.debug_print:
            ones = tf.Print(ones, [shock_event], summarize=100, message='Random shock event: ')
            ones = tf.Print(ones, [shock_time], summarize=100, message='Random shock times: ')
            ones = tf.Print(ones, [shock_multiplier], summarize=100, message='Random shock mults: ')
        shock_multiplier = tf.where(tf.equal(shock_event, 1), x=shock_multiplier, y=ones)
        volume_curves = self.volume_shock(volume_curves=volume_curves,
                                          shock_time=shock_time[:, self._v_idx:self._v_idx+1],
                                          volume_multiplier=shock_multiplier[:, self._v_idx:self._v_idx+1])
        volume_curves = self.bid_shock(volume_curves=volume_curves,
                                       shock_time=shock_time[:, self._b_idx:self._b_idx+1],
                                       bid_multiplier=shock_multiplier[:, self._b_idx:self._b_idx+1])
        return volume_curves

    def volume_shock(self, volume_curves, shock_time, volume_multiplier):
        time_condition = self.time[tf.newaxis, :] >= shock_time
        volume_multipliers = tf.where(time_condition,
                                      x=volume_multiplier * self.ones[tf.newaxis, :],
                                      y=tf.ones_like(time_condition, dtype=self.dtype), name='volume_multipliers')
        return volume_curves * volume_multipliers[:, :, tf.newaxis]

    def bid_shock_old(self, volume_curves, shock_time, bid_multiplier):
        time_condition = self.time[tf.newaxis, :] >= shock_time
        bid_multipliers = tf.where(time_condition,
                                   x=bid_multiplier * self.ones[tf.newaxis, :],
                                   y=tf.ones_like(time_condition, dtype=self.dtype), name='bid_multipliers')

        def bid_shift_single_curve(elem):
            volume_curve, bid_multiplier = elem
            return DataAugmentation.bid_multiplication(self.bid_scale, volume_curve, bid_multiplier)

        def bid_shift(elem):
            volume_curve, bid_multiplier = elem
            return tf.map_fn(fn=bid_shift_single_curve,
                             elems=(volume_curve, bid_multiplier),
                             name='bid_shocked_volume_curves_map_fn', dtype=self.dtype)
        return tf.map_fn(fn=bid_shift, elems=(volume_curves, bid_multipliers[:, :, tf.newaxis]),
                         name='bid_shocked_volume_curves', dtype=self.dtype)

    def bid_shock(self, volume_curves, shock_time, bid_multiplier):
        time_condition = self.time[tf.newaxis, :] >= shock_time
        bid_multipliers = tf.where(time_condition,
                                   x=bid_multiplier * self.ones[tf.newaxis, :],
                                   y=tf.ones_like(time_condition, dtype=self.dtype), name='bid_multipliers')
        return DataAugmentation.bid_multiplication_batch(self.bid_scale,
                                                         volume_curves, bid_multipliers[:, :, tf.newaxis], self.dtype)

    @staticmethod
    def bid_multiplication_batch(bid_scale, volume_curves, bid_multiplier, dtype):
        bid_scale = tf.constant(bid_scale, dtype=dtype)
        m_scale = bid_scale * bid_multiplier
        bid_scale = tf.ones_like(volume_curves, dtype=dtype) * bid_scale
        idx = tf.searchsorted(bid_scale, m_scale, side='right')-1
        volumes = utils.tf_diff_axis_2(volume_curves, name="volumes")
        # bincount naturally works only on a single dimension,
        #  we use a trick with idx_shift and reshape to treat all dimensions at once
        vc_shape = tf.shape(volume_curves)
        idx_shift = tf.range(0, vc_shape[0] * vc_shape[1])
        idx_shift = tf.reshape(idx_shift, [-1, 1])
        idx_shift = tf.tile(idx_shift, multiples=[1, vc_shape[2]]) * vc_shape[2]
        idx_shift = tf.reshape(idx_shift, [vc_shape[0], vc_shape[1], vc_shape[2]])
        minlength = vc_shape[0]*vc_shape[1]*vc_shape[2]
        bincounts = tf.bincount(idx+idx_shift, weights=volumes, minlength=minlength)
        bincounts = tf.reshape(bincounts, vc_shape)
        return tf.cumsum(bincounts, axis=2)

    @staticmethod
    def bid_multiplication(bid_scale, volume_curve, bid_multiplier):
        m_scale = bid_scale * bid_multiplier
        idx = tf.reduce_sum(tf.cast(bid_scale[:, np.newaxis] <= m_scale, tf.int32), axis=0) - 1
        volumes = utils.tf_diff_axis_0(volume_curve, name="volumes")
        return tf.cumsum(tf.bincount(idx, weights=volumes, minlength=len(bid_scale)))
