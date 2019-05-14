from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import glob
import os
import re
from collections import namedtuple
from contextlib import closing
from itertools import groupby
from multiprocessing.pool import Pool
from typing import Iterator

import numpy as np
import pandas as pd
import tensorflow as tf

from rnn_controller.constants import BID_SCALE_EXP, GRID_SIZE_TIME
from rnn_controller.experiment import Experiment

TEST_DATES = ['2018-09-13', '2018-09-14', '2018-10-19', '2018-10-20', '2018-10-21', '2018-10-22', '2018-10-23',
              '2018-10-24']


class Evaluation(namedtuple("EvaluationBase", ("run_name", "step", "value", "checkpoint_path"))):

    ROOT = "/tmp/tensorflow/"
    EVAL_DIR = "eval_test"
    VALUE = "mean_stopped_cost_and_penalty"
    BEST_DIR = "best"
    CHECKPOINT_PREFIX = "model.ckpt"
    EVENT_PREFIX = "events.out.tfevents.*"

    @classmethod
    def get_best_evaluations(cls):
        model_evaluation = {}
        for root, dirs, files in os.walk(cls.ROOT):
            if root.endswith(cls.EVAL_DIR):
                best_eval = best_evaluation(root)
                if not best_eval.is_corrupted():
                    model_evaluation[root] = best_eval
        return [(model_evaluation[n]) for n in sorted(model_evaluation, key=lambda m: model_evaluation[m].value)]

    @classmethod
    def best_dir_path(cls, file: str):
        ckpt_dir, ckpt_name = os.path.split(file)
        return os.path.normpath(os.path.join(ckpt_dir, cls.BEST_DIR, ckpt_name))

    @staticmethod
    def build_from_events(events, run_name):
        if len(events) == 0:
            return Evaluation.corrupted(run_name)
        value = np.nan
        checkpoint_path = ''
        for event in events:
            for v in event.summary.value:
                if v.tag == Evaluation.VALUE:
                    value = v.simple_value
                elif v.tag == "checkpoint_path":
                    checkpoint_path = v.tensor.string_val
                    if not isinstance(checkpoint_path, str):
                        checkpoint_path = checkpoint_path[0]
        return Evaluation(run_name=run_name, step=event.step, value=value, checkpoint_path=checkpoint_path)

    @staticmethod
    def corrupted(run_name):
        return Evaluation(run_name, np.nan, np.nan, "")

    def is_corrupted(self):
        return np.isnan(self.step) or np.isnan(self.value) or len(self.checkpoint_path) == 0

    def best_checkpoint_path(self):
        return os.path.normpath(os.path.join(self.ROOT,
                                             self.run_name.replace(self.EVAL_DIR, ""),
                                             f"model.ckpt-{self.step}"))

    def build_experiment(self, volume_pattern_provider_fn=None, bid_scale=BID_SCALE_EXP, nb_timestamps=GRID_SIZE_TIME):
        checkpoint_dir = self.run_name.replace(self.EVAL_DIR, '')

        model_params = {'bid_scale': bid_scale, 'summarize_gradients': True, }

        controller_params = {"total_time": nb_timestamps, "volume_pattern_provider": volume_pattern_provider_fn, }

        control_params = {"dtype": tf.float64}
        # control_params = {"dtype": tf.float64, 'max_bid_value':10}
        # TODO(at): allow max_bid_value to be set here or through the name parsing
        regex = re.match(r".*regL1_(\d+\.\d*)_L2.*", checkpoint_dir)
        if regex is not None:
            reg_val = float(regex.group(1))
            regularizer = tf.keras.regularizers.l1_l2(l1=reg_val, l2=reg_val)
            control_params = {**control_params,
                              'kernel_regularizer': regularizer,
                              'recurrent_regularizer': regularizer,
                              'bias_regularizer': regularizer, }

        return Experiment.build_from_path(checkpoint_dir, model_params, control_params, controller_params)

    def full_eval(self, campaign_variables=((100, 10), ), volume_pattern_provider_fn=None, test_dates=None,
                  common_data_params=None, use_cache=True):
        experiment, run_name = self.build_experiment(volume_pattern_provider_fn=volume_pattern_provider_fn)

        if test_dates is None:
            test_dates = TEST_DATES

        results = {}
        [results.update(experiment.eval(
            run_name, test_dates=test_dates, common_data_params=common_data_params,
            checkpoint_path=self.checkpoint_path, volume_target=v,
            missed_imp_penalty=p, use_cache=use_cache)) for v, p in campaign_variables]
        return results


def get_evaluations(event_dir: str) -> Iterator[Evaluation]:
    events = []
    for filename in glob.glob(os.path.normpath(os.path.join(event_dir, Evaluation.EVENT_PREFIX))):
        try:
            for i, e in enumerate(tf.train.summary_iterator(filename)):
                events.append(e)
        except tf.errors.DataLossError:
            print(f"{filename} - event {i} is corrupted.")
    events = sorted(events, key=lambda e: e.step)
    run_name = event_dir.replace(Evaluation.ROOT, "")
    return filter(lambda e: not e.is_corrupted(),
                  [Evaluation.build_from_events(list(g), run_name) for k, g in groupby(events, lambda e: e.step)])


def best_evaluation(event_dir: str) -> Evaluation:
    return min(get_evaluations(event_dir=event_dir),
               key=lambda e: e.value,
               default=Evaluation.corrupted(run_name=event_dir.replace(Evaluation.ROOT, "")))


def results_to_df(results):
    data = [list(k) + list(v.values()) for k, v in results.items()]
    columns = [list(k._fields) + list(v.keys()) for k, v in results.items()][0]
    return pd.DataFrame(data, columns=columns)


def add_data_to_df(df, training_date, drop_training=False):
    if isinstance(training_date, str):
        training_date = [training_date]
    df["training"] = df.date.isin(training_date)
    if drop_training:
        df = df[~df["training"]].copy(deep=True)
    r = df[df.control_type == "pi"].groupby("volume_target").mean()['mean_stopped_cost_and_penalty']

    def f(x): return r[x]
    df['cost_penalty_ratio'] = df["mean_stopped_cost_and_penalty"] / df["volume_target"].apply(f)
    df['penalty_ratio'] = df["mean_penalty"] / df["volume_target"].apply(f)
    df['cost_ratio'] = df["mean_stopped_cost"] / df["volume_target"].apply(f)
    return df


@functools.lru_cache(maxsize=128, typed=False)
def single_eval(run_name, step, value, checkpoint_path, campaign_variable, test_date,
                volume_pattern_provider_fn=None, common_data_params=None, use_cache=True):
    evaluation = Evaluation(run_name=run_name, step=step, value=value, checkpoint_path=checkpoint_path)
    if hasattr(common_data_params, "_asdict"):
        common_data_params = common_data_params._asdict()
    return evaluation.full_eval(campaign_variables=[campaign_variable],
                                volume_pattern_provider_fn=volume_pattern_provider_fn,
                                test_dates=[test_date], common_data_params=common_data_params, use_cache=use_cache
                                )


DataParams = namedtuple("DataParams", ["suffix", "volume_threshold", "batch_size", "grid_size_bid"])


def parallel_eval(evaluations, campaign_variables=((100, 10), ), test_dates=None, volume_pattern_provider_fn=None,
                  common_data_params=None, use_cache=True):
    if test_dates is None:
        test_dates = TEST_DATES
    # TODO(nperrin16): replace by *e?
    arguments = [(e.run_name, e.step, e.value, e.checkpoint_path, cv, test_date, volume_pattern_provider_fn,
                  common_data_params, use_cache)
                 for e in evaluations
                 for cv in campaign_variables
                 for test_date in test_dates]
    with closing(Pool()) as pool:
        results = pool.starmap(single_eval, arguments)
    return {k: v for d in results for k, v in d.items()}
