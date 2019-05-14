from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from rnn_controller.constants import FLOAT_TYPE, MIN_DAILY_VOLUME, GRID_SIZE_BID_EXP, GRID_SIZE_TIME
from rnn_controller.data import get_day_dataset, get_all_tags, fill_cache


class VolumePattern(tf.keras.layers.Layer):
    """Layer (1 + c1*sin(2*pi*t/nb_timestamps + phi1) + c2*sin(4*pi*t/nb_timestamps + phi2))/nb_timestamps
    """
    def __init__(self, nb_timestamps=GRID_SIZE_TIME, c1=1, phi1=0, c2=1, phi2=0,
                 trainable=True, name=None, dtype=FLOAT_TYPE, **kwargs):
        self.nb_timestamps = nb_timestamps
        self.time = np.linspace(0, 1, num=nb_timestamps)[np.newaxis, :]
        self.c1 = c1
        self.c2 = c2
        self.phi1 = phi1
        self.phi2 = phi2
        super().__init__(trainable, name, dtype, **kwargs)

    def build(self, input_shape):
        self.c1 = self.add_weight(name='c1', shape=(), dtype=self.dtype,
                                  initializer=tf.constant_initializer(self.c1, dtype=self.dtype))
        self.phi1 = self.add_weight(name='phi1', shape=(), dtype=self.dtype,
                                    initializer=tf.constant_initializer(self.phi1, dtype=self.dtype))
        self.c2 = self.add_weight(name='c2', shape=(), dtype=self.dtype,
                                  initializer=tf.constant_initializer(self.c2, dtype=self.dtype))
        self.phi2 = self.add_weight(name='phi2', shape=(), dtype=self.dtype,
                                    initializer=tf.constant_initializer(self.phi2, dtype=self.dtype))
        self.time = tf.constant(self.time, shape=(1, self.nb_timestamps), dtype=self.dtype)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pattern = 1. + self.c1 * tf.sin(2 * np.pi * self.time + self.phi1, name='sin_pattern1') \
               + self.c2 * tf.sin(4 * np.pi * self.time + self.phi2, name='sin_pattern2')
        pattern = pattern / self.nb_timestamps
        return tf.tile(pattern, multiples=(tf.shape(inputs)[0], 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.nb_timestamps

    def pattern_provider_fn(self):
        def pattern_provider(current_time):
            pattern = 1. + self.c1 * tf.sin(2 * np.pi * self.time + self.phi1, name='sin_pattern1') \
                   + self.c2 * tf.sin(4 * np.pi * self.time + self.phi2, name='sin_pattern2')
            # As we later on compute a ratio, we don't renormalize to avoid numerical issues.
            pattern = tf.tile(pattern, multiples=(tf.shape(current_time)[0], 1))  # / self.nb_timestamps
            remaining_pattern = tf.where(condition=self.time >= current_time / self.nb_timestamps,
                                         x=pattern,
                                         y=tf.zeros_like(pattern, dtype=self.dtype))
            return tf.reduce_sum(remaining_pattern, axis=-1, keepdims=True)
        return pattern_provider


class VolumePatternEstimation(object):
    DATA_PARAMS = {'date1': "2018-09-13", 'suffix': '', 'volume_threshold': MIN_DAILY_VOLUME,
                   'grid_size_bid': GRID_SIZE_BID_EXP}
    LEARNING_RATE = 1e-1
    DEFAULT_PARAMS = {'c1': 0.567, 'c2': 0.277, 'phi1': -1.1894, 'phi2': -0.25,
                      'nb_timestamps': GRID_SIZE_TIME,
                      'learning_rate': LEARNING_RATE}
    MODEL_DIR = "/tmp/tensorflow_models/volume_pattern_estimation"

    def __init__(self, params, data_params=None, model_dir=MODEL_DIR, run_config=None, dtype=FLOAT_TYPE):
        if data_params is None:
            data_params = self.DATA_PARAMS.copy()
        self._data_params = data_params
        self._params = params
        self._run_config = run_config
        self._model_dir = model_dir
        self.dtype = dtype

    def estimator(self):
        return tf.estimator.Estimator(model_fn=self.model_fn, params=self.add_default_params(self._params),
                                      model_dir=self._model_dir, config=self._run_config)

    def input_fn(self, tags, batch_size, data_params=None, take_count=-1, repeat_count=None,
                 cache_dir=None, prefetch_buffer_size=None):
        params = self._data_params if data_params is None else data_params
        label_dataset = get_day_dataset(tags, **params, batch_size=batch_size,
                                        take_count=take_count, repeat_count=repeat_count,
                                        cache_directory_root=cache_dir, prefetch_buffer_size=prefetch_buffer_size)
        # take the total volume of each snapshot and renormalize by the day's total volume
        label_dataset = label_dataset\
            .map(lambda v: v[:, :, v.shape[2]-1])\
            .map(lambda v: v / tf.reduce_sum(v, axis=1, keepdims=True))
        # feature is a dummy tensor with the correct shape
        feature_dataset = tf.data.Dataset.from_tensors(tf.zeros((1,), dtype=self.dtype)) \
            .repeat().batch(batch_size)
        return tf.data.Dataset.zip((feature_dataset, label_dataset))

    def create_model(self, params):
        return VolumePattern(params['nb_timestamps'], params['c1'], params['c2'], params['phi1'], params['phi2'])

    def model_fn(self, features, labels, mode, params):
        """The model_fn argument for creating an Estimator.
            features: empty dict
            labels: the input volume curves renormalized (volume patterns)
        """
        # Define the RNN
        model = self.create_model(params)
        volume_pattern = model(features)  # just need the shape
        volume_pattern = tf.Print(volume_pattern, [model.c1, model.c2, model.phi1, model.phi2], '[c1, c2, phi1, phi2]:')

        tf.summary.scalar('volume_pattern/c1', model.c1)
        tf.summary.scalar('volume_pattern/c2', model.c2)
        tf.summary.scalar('volume_pattern/phi1', model.phi1)
        tf.summary.scalar('volume_pattern/phi2', model.phi2)

        # Handle model operations
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'volume_pattern': volume_pattern,
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions)

        loss = tf.losses.mean_squared_error(labels, volume_pattern)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], name="optimizer")

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step()))

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'rmse':
                        tf.metrics.root_mean_squared_error(labels=labels, predictions=volume_pattern),
                })

    def add_default_params(self, params):
        d = self.DEFAULT_PARAMS.copy()
        d.update(params)
        return d


def main(flags, run_params, data_params, save_plot_callback=None, saving_listeners=None):
    tags = get_all_tags(date1=data_params['date1'], suffix=data_params['suffix'])
    print("Number of tags:", len(tags))
    params = {}
    batch_size = run_params['batch_size']
    run_config = tf.estimator.RunConfig(log_step_count_steps=run_params['log_step_count_steps'],
                                        save_summary_steps=run_params['save_summary_steps'])
    volume_pattern_estimation = VolumePatternEstimation(params, data_params,
                                                        run_config=run_config, model_dir=flags.checkpoint_dir)

    with tf.Session() as sess:
        fill_cache(sess, volume_pattern_estimation.input_fn(tags, batch_size,
                                                            repeat_count=1, cache_dir=flags.cache_dir))

    def train_input_fn():
        return volume_pattern_estimation.input_fn(tags, batch_size=batch_size, cache_dir=flags.cache_dir)

    estimator = volume_pattern_estimation.estimator()
    print("Estimator.config: log_step_count_steps=", estimator.config.log_step_count_steps,
          ", save_summary_steps=", estimator.config.save_summary_steps)

    estimator = estimator.train(input_fn=train_input_fn, steps=run_params.get('train_steps', None),
                                max_steps=run_params.get('max_train_steps', None),
                                saving_listeners=saving_listeners)

    train_results = estimator.evaluate(input_fn=train_input_fn, steps=1, name='train')
    print("Train:", train_results)
    try:
        print("Variable names:", estimator.get_variable_names())
        for v_name in ['c1', 'c2', 'phi1', 'phi2']:
            print(v_name + " is:", estimator.get_variable_value('volume_pattern/' + v_name))
    except tf.errors.NotFoundError:
        print("Trained model checkpoint not yet available")

    pred_result = next(estimator.predict(input_fn=lambda: train_input_fn().take(1)))
    volume_pattern_val = pred_result['volume_pattern']
    plt.plot(volume_pattern_val, label='model')


if __name__ == "__main__":
    import rnn_controller.tools as tools
    import logging

    tf.logging.set_verbosity(tf.logging.INFO)

    default_run_params = {'batch_size': 8,
                          'log_step_count_steps': 10,
                          'save_summary_steps': 10,
                          'train_steps': 100}

    args = tools.Struct()
    args.run_name = 'test'
    args.restart = True
    args.run_params = default_run_params
    flags = tools.initialize(args.run_name, restart=args.restart)
    print(flags)

    date1 = "2018-09-03"
    date2 = "20180903"
    data_params = {'date1': date1, 'date2': date2, 'suffix': '', 'volume_threshold': 300}
    main(flags, data_params=data_params, run_params=args.run_params)
