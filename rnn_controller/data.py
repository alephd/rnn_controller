from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import hashlib
import inspect
from collections import namedtuple

import tensorflow as tf
import tensorflow_probability as tfp

from rnn_controller.constants import FLOAT_TYPE, INT_TYPE, MIN_DAILY_VOLUME, CACHE_DIRECTORY, SHUFFLE_SEED, \
    TF_RANDOM_SEED

DEFAULT_SUFFIX = '_small'

bucketname = "/tmp"
path = "noisy_controller_demo{suffix}"
dirname_format = "{bucketname}/{path}/{date1}/"
filename_format = dirname_format + "{date2}_{tag}"


def _parse_function(grid_size_bid):
    def parse_function(sequence_example_proto):
        sequence_features = {'timeframes': tf.FixedLenSequenceFeature(shape=[grid_size_bid], dtype=INT_TYPE)}
        _, sequence = tf.parse_single_sequence_example(sequence_example_proto, sequence_features=sequence_features)
        return sequence["timeframes"]
    return parse_function


@functools.lru_cache(maxsize=128, typed=False)
def get_all_tags(date1, suffix):
    """Returns list of tags for a given date."""
    assert len(suffix) == 0 or suffix == "_small"
    filenames = tf.gfile.ListDirectory(dirname_format.format(bucketname=bucketname, path=path.format(suffix=suffix),
                                                             date1=date1))
    return [f.split('_')[-1] for f in filenames if f != '_SUCCESS']


def get_filename_for_tag(tag, date1, suffix):
    """Returns the filename associated to a tag for a given date."""
    date2 = date1.replace("-", "")
    return filename_format.format(bucketname=bucketname, path=path.format(suffix=suffix), date1=date1, date2=date2,
                                  tag=tag)


def get_snapshot_dataset_for_tag(tag, date1, suffix, grid_size_bid):
    """Returns a unbatched `tf.dataTFRecordDataset` for given tag and date.
        dataset iterator get_next shape is (GRID_SIZE_BID,)
    """
    dataset = get_day_dataset_for_tag(tag, date1, suffix=suffix, grid_size_bid=grid_size_bid)
    return dataset.apply(tf.contrib.data.unbatch())


def get_day_dataset_for_tag(tag, date1, suffix, grid_size_bid):
    """Returns a `tf.dataTFRecordDataset` for given tag and date.
        dataset iterator get_next shape is (GRID_SIZE_TIME,GRID_SIZE_BID)
    """
    dataset = tf.data.TFRecordDataset([get_filename_for_tag(tag, date1, suffix)])
    return dataset.map(_parse_function(grid_size_bid)).map(lambda p: tf.cast(p, FLOAT_TYPE))


def get_unbatched_day_dataset(tags, date1, suffix, grid_size_bid, volume_threshold=MIN_DAILY_VOLUME, take_count=-1,
                              repeat_count=None, num_parallel_reads=2, shuffle_buffer_size=None,
                              prefetch_buffer_size=1000, cache_directory_root=CACHE_DIRECTORY):
    """Returns a `tf.dataTFRecordDataset` for given tags and date. Filtering out tags with less than volume_threshold.
        dataset iterator get_next shape is (GRID_SIZE_TIME, GRID_SIZE_BID)
    """
    dataset = tf.data.TFRecordDataset([get_filename_for_tag(tag, date1, suffix) for tag in tags],
                                      num_parallel_reads=num_parallel_reads)

    def predicate(volume_curve):
        return tf.greater_equal(tf.reduce_sum(volume_curve[:, grid_size_bid-1], 0), volume_threshold)

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)\
        .map(_parse_function(grid_size_bid))\
        .map(lambda p: tf.cast(p, FLOAT_TYPE))\
        .filter(predicate)\
        .take(count=take_count)
    if cache_directory_root is not None:
        arg_info = inspect.getargvalues(inspect.currentframe())
        cached_args = sorted([k for k in arg_info.args
                              if k in ['tags', 'date1', 'suffix', 'grid_size_bid', 'volume_threshold', 'take_count',
                                       'num_parallel_reads']])
        args = [str(k) for k in zip(cached_args, [arg_info.locals[k] for k in cached_args])]
        hashed_args = hashlib.md5(str(args).encode()).hexdigest()
        cache_directory = cache_directory_root + hashed_args + "/"
        print("Using cache_directory {}".format(cache_directory))
        tf.logging.log(tf.logging.INFO, "Using cache_directory %s", cache_directory)
        tf.gfile.MakeDirs(cache_directory)
        dataset = dataset.cache(cache_directory)
    if shuffle_buffer_size is not None:
        # https://www.tensorflow.org/performance/datasets_performance#repeat_and_shuffle
        return dataset.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size, count=repeat_count,
                                                                     seed=SHUFFLE_SEED))
    return dataset.repeat(count=repeat_count)


def get_day_dataset(tags, date1, suffix, grid_size_bid, volume_threshold=MIN_DAILY_VOLUME, batch_size=None,
                    num_parallel_reads=2, prefetch_buffer_size=None, take_count=-1, repeat_count=None,
                    shuffle_buffer_size=None, cache_directory_root=CACHE_DIRECTORY):
    """Returns a `tf.dataTFRecordDataset` for given tags and date. Filtering out tags with less than volume_threshold.
        dataset iterator get_next shape is (batch_size, GRID_SIZE_TIME, GRID_SIZE_BID)
        If batch_size is None (default) : batch_size used is len(tags)
    """
    if not batch_size:
        batch_size = len(tags)
    if not prefetch_buffer_size:
        prefetch_buffer_size = batch_size
    with tf.name_scope("data_processing"):
        return get_unbatched_day_dataset(tags, date1, suffix, grid_size_bid=grid_size_bid,
                                         volume_threshold=volume_threshold, take_count=take_count,
                                         repeat_count=repeat_count, num_parallel_reads=num_parallel_reads,
                                         shuffle_buffer_size=shuffle_buffer_size,
                                         prefetch_buffer_size=prefetch_buffer_size,
                                         cache_directory_root=cache_directory_root) \
            .batch(batch_size)


def get_days_dataset(tags, dates, suffix, grid_size_bid, volume_threshold=MIN_DAILY_VOLUME, batch_size=None,
                     num_parallel_reads=2, prefetch_buffer_size=None, take_count=-1, repeat_count=None,
                     shuffle_buffer_size=None, cache_directory_root=CACHE_DIRECTORY):
    """Returns a `tf.dataTFRecordDataset` for given tags and date. Filtering out tags with less than volume_threshold.
        dataset iterator get_next shape is (batch_size, GRID_SIZE_TIME, GRID_SIZE_BID)
        If batch_size is None (default) : batch_size used is len(tags)
    """
    if not batch_size:
        batch_size = len(tags)
    if not prefetch_buffer_size:
        prefetch_buffer_size = batch_size
    with tf.name_scope("data_processing"):
        datasets = []
        for date in dates:
            tags = get_all_tags(date, suffix)
            datasets.append(get_unbatched_day_dataset(tags, date, suffix, grid_size_bid=grid_size_bid,
                                                      volume_threshold=volume_threshold, take_count=take_count,
                                                      repeat_count=repeat_count, num_parallel_reads=num_parallel_reads,
                                                      shuffle_buffer_size=shuffle_buffer_size,
                                                      prefetch_buffer_size=prefetch_buffer_size,
                                                      cache_directory_root=cache_directory_root))
        return tf.data.experimental.sample_from_datasets(datasets, seed=TF_RANDOM_SEED).batch(batch_size)


def get_total_volume(session, tag, date1, suffix, grid_size_bid):
    """Returns total volume for given tag and date."""
    dataset = get_day_dataset_for_tag(tag, date1, suffix=suffix, grid_size_bid=grid_size_bid)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    total_volume = tf.reduce_sum(next_element[:, grid_size_bid-1], 0)
    total_volume_val = session.run(total_volume)
    return total_volume_val


def fill_cache(session, dataset, log_every_n=10):
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    i = 0
    while True:
        i += 1
        try:
            session.run(next_element)
        except tf.errors.OutOfRangeError:
            break
        if log_every_n is not None and i % log_every_n == 0:
            print("Cached dataset elements: {}".format(i))
    print("Done caching")


def count_batches(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    i = 0
    with tf.Session() as session:
        while True:
            try:
                session.run(next_batch)
                i += 1
            except tf.errors.OutOfRangeError:
                break
    return i


class CampaignSpec(namedtuple("CampaignSpecBase", ("volume_target", "missed_imp_penalty"))):
    pass


class CampaignInitializer(namedtuple("VolumeTargetBase",
                                     ["volume_target_initializer", "missed_imp_penalty_initializer", "absolute", "seed"]
                                     )):

    def add_campaign_data(self, batched_dataset):
        """Add campaign data based on sampling to the batched dataset.

        In the non-absolute mode, volume target samples are the proportion of the batch mean total volume.

        Args:
            batched_dataset:

        Returns:
            `tf.data.Dataset`  zip(batched_dataset, campaign_data)

        """
        def _add_campaign_data(volume_curves):
            with tf.name_scope("add_campaign_data"):
                volume_target_samples = self.volume_target_initializer(volume_curves.dtype).sample(
                    seed=self.seed, name="volume_target_samples")
                penalty = self.missed_imp_penalty_initializer(volume_curves.dtype).sample(
                    seed=self.seed, name="penalty")
                if self.absolute:
                    volume_target = tf.identity(volume_target_samples, name="volume_target")
                else:
                    max_volumes = tf.identity(tf.reduce_sum(volume_curves[:, :, -1], axis=1), name="max_volumes")
                    # mean_volume = tf.reduce_mean(max_volumes, name="batch_mean_volume")
                    median_volume = tfp.distributions.percentile(max_volumes, 50.0, name="batch_median_volume")
                    volume_target = tf.identity(volume_target_samples * median_volume, name="volume_target")

                return volume_curves, {"volume_target": volume_target, "missed_imp_penalty": penalty}
        return batched_dataset.map(_add_campaign_data)

    @staticmethod
    def build_deterministic(volume_target, missed_imp_penalty, absolute, seed):
        return CampaignInitializer(
            lambda dtype: tfp.distributions.Deterministic(tf.convert_to_tensor(volume_target, dtype=dtype)),
            lambda dtype: tfp.distributions.Deterministic(tf.convert_to_tensor(missed_imp_penalty, dtype=dtype)),
            absolute=absolute, seed=seed)

    @staticmethod
    def build_deterministic_from_spec(campaign_eval_spec: CampaignSpec, absolute: bool, seed):
        return CampaignInitializer(
            lambda dtype: tfp.distributions.Deterministic(
                tf.convert_to_tensor(campaign_eval_spec.volume_target, dtype=dtype)),
            lambda dtype: tfp.distributions.Deterministic(
                tf.convert_to_tensor(campaign_eval_spec.missed_imp_penalty, dtype=dtype)),
            absolute=absolute, seed=seed)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from rnn_controller.constants import GRID_SIZE_BID_EXP

    date1 = '2018-09-13'
    tags = get_all_tags(date1=date1, suffix=DEFAULT_SUFFIX)
    assert tags[0] == '1028526'

    dataset = get_day_dataset_for_tag(tags[0], date1=date1, suffix=DEFAULT_SUFFIX, grid_size_bid=GRID_SIZE_BID_EXP)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        element_val = sess.run(next_element)
        plt.plot(element_val[:, -1], '-b')
        plt.plot(element_val[:, -50], '-r')
        plt.show()
