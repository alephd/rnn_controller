from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re
from collections import namedtuple

import numpy as np
import tensorflow as tf

import rnn_controller.tools as tools
from rnn_controller import utils
from rnn_controller.actuator import create_actuator
from rnn_controller.constants import FLOAT_TYPE, GRID_SIZE_BID_EXP, MIN_BID_VALUE, MIN_NOISE_VALUE, TF_RANDOM_SEED
from rnn_controller.control import create_control, PIControlCell, PIDControlCell
from rnn_controller.controller import ControllerCell, ControllerRNNCell
from rnn_controller.data import get_day_dataset, fill_cache, get_all_tags, CampaignInitializer, get_days_dataset
from rnn_controller.model import ControllerModel, LR_SUMMARY_KEY, GRAD_SUMMARY_KEY, LOSS_SUMMARY_KEY
from rnn_controller.optimizer import ScheduledSGDOptimizer, GenericOptimizer, AnnealedSGDOptimizer
from rnn_controller.strategy import compute_stopped_results, compute_penalty
from rnn_controller.toy_volume_curves import DataAugmentation
from rnn_controller.utils import parse_eval_events

ExperimentRunKey = namedtuple("ExperimentRunKey",
                              ["actuator_type", "controller_type", "control_type", "extra_control_input_names",
                               "relative", "optimizer_name", "run_name", "date", "volume_target", "missed_imp_penalty",
                               "path"])


class Experiment(object):

    def __init__(self, params, control_params, controller_params, control_type, dtype=FLOAT_TYPE,
                 dir_initializer=None):
        self._params = params
        self._control_type = control_type
        self._control_params = control_params
        self._controller_params = controller_params
        self.dtype = dtype
        tf.keras.backend.set_floatx(dtype.name)
        if dir_initializer is None:
            dir_initializer = tools.DirInitializer()
        self._dir_initializer = dir_initializer
        self._optimizer = None

    @staticmethod
    def build_from_path(model_path, params, control_params, controller_params):

        experiment_args, run_name = parse(model_path)
        params = params.copy()
        params.update(experiment_args["params"])
        experiment_args["params"] = params
        _default_update(experiment_args, "controller_params", controller_params)

        if experiment_args["params"].get("controller_type", "") == "rnn":
            try:
                del experiment_args["controller_params"]['volume_pattern_provider']
            except KeyError:
                pass

        if experiment_args["control_type"].startswith("constant_noise"):
            if "_NOISE_" in run_name:
                noise_level = float(run_name.split("_NOISE_")[-1].replace("/", ""))
                initial_state_params = (noise_level, False,
                                        {'constraint': lambda noise: tf.clip_by_value(noise, MIN_NOISE_VALUE, 1.)})
            else:
                initial_state_params = (0.2, True, {'constraint': lambda noise: tf.clip_by_value(noise, 0.01, 1.)})
            constant_noise_params = {"initial_state_params": initial_state_params}
            control_params["constant_noise_params"] = constant_noise_params
        _default_update(experiment_args, "control_params", control_params)
        print(f"Building experiment with: {experiment_args}")
        return Experiment(**experiment_args), run_name

    @property
    def actuator_type(self):
        return self._params["actuator_type"].replace("_actuator", "").replace("_wrapper", "")

    @property
    def controller_type(self):
        return self._params.get("controller_type", "")

    @property
    def control_type(self):
        return self._control_type.replace("_control", "")

    @property
    def extra_control_input_names(self):
        return self._controller_params.get("extra_control_input_names", tuple())

    @property
    def relative(self):
        return self._controller_params.get("relative", False)

    @property
    def is_pi(self):
        return self._control_type.startswith("pi") and not self.is_pid

    @property
    def is_pid(self):
        return self._control_type.startswith("pid")

    @property
    def is_rnn(self):
        return self.controller_type == "rnn"

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer_name(self):
        return self._params["optimizer_name"]

    @property
    def regularization(self):
        return self._params.get("regularization", 0)

    @property
    def spread_reg(self):
        return self._params.get("spread_reg", 0)

    @property
    def model_dir(self):
        extra_map = {'cost': 'c', 'volume': 'v', 'volume_target': 'vt', "time_embedding": "te"}
        model_dirs = [
            self.actuator_type,
            self.controller_type,
            self.control_type,
            "_".join([extra_map[s] for s in self.extra_control_input_names]),
            "relative" if self.relative else "",
        ]
        model_param = []
        if "initial_state_params" in self._control_params:
            init_state = self._control_params["initial_state_params"]
            r = []
            for value in init_state:
                r.append(str([v.tolist() if type(v) == np.ndarray else v
                              for e, v in enumerate(value) if e < 2]).replace("False", "F")
                         .replace("True", "T")
                         .replace(", ", "_")
                         .replace("[", "")
                         .replace("]", "")
                         .replace("(", "")
                         .replace(")", "")
                         .replace("array", "")
                         .replace(" ", ""))
            model_param.append("ctrli_{}".format("__".join(r)))
        if self.is_pi or self.is_pid:
            # "iT_k1_ti200_pg1e3_benchmark"
            # 'k_l': 1, 't_i': 200, 't_s': 5, 'plant_gain_var': 1e3, 'use_log': True,
            k_l = self._control_params['k_l']
            t_i = self._control_params['t_i']
            t_s = self._control_params['t_s']
            plant_gain_var = self._control_params.get('plant_gain_var', 1.)
            use_log = self._control_params.get('use_log', True)
            model_param.append(f"PI{'D' if self.is_pid else ''}_k{k_l:g}_ti{t_i:g}_ts{t_s:g}_pg{plant_gain_var:g}_log{'T' if use_log else 'F'}")
        return os.path.normpath("/".join(model_dirs+model_param))

    @property
    def run_name_prefix(self):
        return get_run_name_prefix(self._params)

    @property
    def cache_dir(self):
        return self._dir_initializer.cache_dir

    def initialize_dir(self, run_name, restart):
        return self._dir_initializer(self.get_run_dir(run_name), restart=restart)

    def get_run_dir(self, run_name):
        return os.path.normpath(os.path.join(
            self.model_dir,
            self.optimizer_name,
            self.run_name_prefix + run_name))

    def get_run_key(self, run_name, date, volume_target, missed_imp_penalty):
        return ExperimentRunKey(
            self.actuator_type, self.controller_type, self.control_type, self.extra_control_input_names, self.relative,
            self.optimizer_name, run_name, date, volume_target, missed_imp_penalty, self.get_run_dir(run_name))

    def estimator(self, model_dir, run_config=None, warm_start_from=None, tf_random_seed=TF_RANDOM_SEED, **params):
        if run_config is None:
            run_config = tf.estimator.RunConfig(tf_random_seed=tf_random_seed,
                                                keep_checkpoint_max=15,
                                                log_step_count_steps=50,
                                                save_summary_steps=10,
                                                save_checkpoints_steps=200)

        return tf.estimator.Estimator(model_fn=self.model_fn, params=params, model_dir=model_dir, config=run_config,
                                      warm_start_from=warm_start_from)

    @staticmethod
    def input_fn(tags, data_params):
        print("Calling experiment input_fn with data_params:", data_params)
        data_params = data_params.copy()
        augmentation_params = data_params.pop('augmentation', None)
        campaign_initializer = data_params.pop("campaign_initializer", None)

        if "date1" in data_params:
            dataset = get_day_dataset(tags, **data_params)
        elif "dates" in data_params:
            dataset = get_days_dataset(tags=None, **data_params)
        else:
            raise ValueError("date1 or dates should be passed into data_params.")

        if augmentation_params is not None:
            augment = DataAugmentation(**augmentation_params)
            dataset = augment(dataset)
        if campaign_initializer is not None:
            dataset = campaign_initializer.add_campaign_data(dataset)
        return dataset

    def create_model(self, params, volume_target, missed_imp_penalty, initial_state=None):
        control_params = self._control_params.copy()
        # TODO(nperrin16): Trainable max bid.
        max_bid_value = control_params.pop("max_bid_value", np.infty if self.is_pi or self.is_pid else missed_imp_penalty)
        if not self.is_pi and not self.is_pid and"activation" not in control_params:
            print("Using default (non-tanh) activation function")

            def activation(control_output):

                def _constraint(x, min_, max_):
                    return (max_ - min_) * tf.tanh(x) / 2. + (min_ + max_) / 2.
                return tf.concat(
                    [_constraint(control_output[:, 0:1], MIN_BID_VALUE, max_bid_value),
                     _constraint(control_output[:, 1:2], MIN_NOISE_VALUE, 1.),
                     tf.tanh(control_output[:, 2:])], axis=1)
            control_params["activation"] = activation

        if "control_output_constraint" not in control_params:
            control_params["control_output_constraint"] = lambda u: utils.control_output_constraint(
                u, max_bid=max_bid_value, min_noise=MIN_NOISE_VALUE, max_noise=1.)
        control_cell = create_control(self._control_type, control_params)
        # TODO: introduce actuator params?
        # WARNING: Don't use the actuator_type property which outputs a human value (e.g. remove `wrapper`)
        actuator_cell = create_actuator(params['actuator_type'], params)
        print(
            f"Creating model using {self.actuator_type} actuator cell{' and RNN controller' if self.is_rnn else ''}"
        )
        if self.is_rnn:
            controller_cell = ControllerRNNCell(
                volume_target=volume_target, cell=control_cell, **self._controller_params)
        else:
            controller_cell = ControllerCell(
                volume_target=volume_target, control=control_cell, **self._controller_params)
        return ControllerModel(controller_cell, actuator_cell, missed_imp_penalty=missed_imp_penalty, dtype=self.dtype,
                               initial_state=initial_state)

    def model_fn(self, features, labels, mode, params, config):
        """Used by the Estimator to create a model.

        Args:
            features: the input volume curves.
            labels: the campaign data {volume target, missed impression penalty,}.
            mode: run mode (PREDICT, TRAIN, EVAL).
            params: {volume target, missed impression penalty,} to use if no labels provided.
            config: not used.

        Returns:
            tf.estimator.EstimatorSpec for the mode.

        """
        if labels is not None:
            print("Using random volume target and missed impression penalty.")
            volume_target = labels["volume_target"]
            missed_imp_penalty = labels["missed_imp_penalty"]
        else:
            print("Using constant volume target and missed impression penalty.")
            volume_target = tf.convert_to_tensor(params["volume_target"], dtype=self.dtype)
            missed_imp_penalty = tf.convert_to_tensor(params["missed_imp_penalty"], dtype=self.dtype)

        model = self.create_model(self._params, volume_target=volume_target, missed_imp_penalty=missed_imp_penalty)
        realized_volume, total_cost, states = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)

        model.add_plot_snapshot(states, self._params.get("image_summary_max_state_size", 1))

        training_loss = model.get_training_loss(self.regularization, self.spread_reg)
        summary_op = tf.summary.merge([tf.summary.merge_all(collection) for collection in [tf.GraphKeys.SUMMARIES,
                                                                                           LOSS_SUMMARY_KEY, ]])
        scaffold = tf.train.Scaffold(summary_op=summary_op)

        # Handle model operations
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            with tf.name_scope("learning_rate_schedule"):
                learning_rate = self._params['learning_rate_schedule'](global_step)
                tf.summary.scalar("learning_rate", learning_rate, collections=[LR_SUMMARY_KEY], family="learning_rates")
                self._optimizer = self._params['optimizer_initializer'](learning_rate=learning_rate,
                                                                        global_step=global_step)
            gradients, variables = model.compute_gradients(
                self.optimizer, training_loss,
                summarize_gradients=self._params.get('summarize_gradients', False),
                clip_norm=self._params.get('clip_norm', None))
            train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            model.summarize_learning_rate(self.optimizer)
            summaries = [tf.summary.merge_all(collection) for collection in [LR_SUMMARY_KEY, GRAD_SUMMARY_KEY]]
            summary_op = tf.summary.merge([summary_op] + [s for s in summaries if s is not None])
            scaffold = tf.train.Scaffold(summary_op=summary_op)

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=training_loss,
                train_op=train_op, scaffold=scaffold)

        with tf.name_scope("eval_stopped_cost"):
            _, _, stopped_cost = compute_stopped_results(states, model.volume_target, sequence=False)
            penalty = compute_penalty(realized_volume, model.volume_target, model.missed_imp_penalty)
            stopped_cost_and_penalty = stopped_cost + penalty
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                **states._asdict(),
                'total_realized_volume': realized_volume,
                'total_cost': total_cost,
                'total_stopped_cost': stopped_cost,
                'total_penalty': penalty,
                'total_stopped_cost_and_penalty': stopped_cost_and_penalty,
            }
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions,
                                              scaffold=scaffold)

        if mode == tf.estimator.ModeKeys.EVAL:
            # Adding random batch eval summary: https://github.com/tensorflow/tensorflow/issues/15332
            # See estimator.eval_dir (we do not have access to name here :s)
            # eval_dir = os.path.join(config.model_dir, 'eval_test')
            # summary_hook = tf.train.SummarySaverHook(save_steps=config.save_summary_steps,
            #                                          output_dir=eval_dir,
            #                                          summary_op=tf.summary.merge_all(key=tools.IMAGE_SUMMARIES_KEY))

            labels = tf.ones_like(realized_volume) * model.volume_target
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=model.loss/model.max_budget,
                # evaluation_hooks=[summary_hook],
                eval_metric_ops={
                    'mean_cost': tf.metrics.mean(total_cost/model.max_budget, name='mean_cost'),
                    'mean_stopped_cost': tf.metrics.mean(stopped_cost/model.max_budget, name='mean_stopped_cost'),
                    'mean_penalty': tf.metrics.mean(penalty/model.max_budget, name='mean_penalty'),
                    'mean_cost_and_penalty': tf.metrics.mean((total_cost + penalty)/model.max_budget,
                                                             name='mean_cost_and_penalty'),
                    'mean_stopped_cost_and_penalty': tf.metrics.mean(stopped_cost_and_penalty/model.max_budget,
                                                                     name='mean_stopped_cost_and_penalty'),
                    'mean_realized_volume': tf.metrics.mean(realized_volume, name='mean_realized_volume'),
                    'mean_target': tf.metrics.mean(model.volume_target, name='mean_realized_volume'),
                    # 'rmse':
                    #     tf.metrics.root_mean_squared_error(labels=labels, predictions=realized_volume),
                    # 'mean_absolute_error':
                    #     tf.metrics.mean_absolute_error(labels=labels, predictions=realized_volume),
                    'mean_relative_volume_error':
                        tf.metrics.mean_relative_error(labels=labels, predictions=realized_volume, normalizer=labels),
                })

    def fill_cache(self, date, data_params):
        # Fill local cache file
        with tf.name_scope("fill_cache"):
            tags = get_all_tags(date1=date, suffix=data_params['suffix'])
            data_params = data_params.copy()
            data_params.update({"date1": date, "batch_size": 10, "prefetch_buffer_size": 100, "repeat_count": 1,
                                "cache_directory_root": self.cache_dir})
            data_params.pop("volume_target", None)
            with tf.Session() as sess:
                fill_cache(sess, self.input_fn(tags, data_params=data_params))

    def run(self, run_name, restart, train_data_params, test_data_params, max_train_steps,
            volume_target=None, missed_imp_penalty=None, run_config=None, warm_start_from=None, eval_spec=None,
            eval_campaign_specs=None,
            tf_random_seed=TF_RANDOM_SEED):
        run_dirs = self.initialize_dir(run_name, restart)

        estimator = self.estimator(model_dir=run_dirs.checkpoint_dir, run_config=run_config,
                                   warm_start_from=warm_start_from, tf_random_seed=tf_random_seed,
                                   volume_target=volume_target, missed_imp_penalty=missed_imp_penalty)

        train_data_params_ = {"repeat_count": 300, "shuffle_buffer_size": 2000, "cache_directory_root": self.cache_dir}
        train_data_params_.update(train_data_params)
        with tf.name_scope("train_tags"):
            if "date1" in train_data_params_:
                train_tags = get_all_tags(date1=train_data_params_["date1"], suffix=train_data_params_.get('suffix', ''))
            else:
                train_tags = None
            train_spec = tf.estimator.TrainSpec(
                lambda: self.input_fn(train_tags, data_params=train_data_params_),
                max_steps=max_train_steps,
                hooks=self.get_hooks(estimator, dirs=run_dirs)
            )

        if eval_spec is None:
            test_data_params_ = test_data_params.copy()
            test_data_params_["repeat_count"] = 1
            test_data_params_["cache_directory_root"] = self.cache_dir
            if eval_campaign_specs is None:
                with tf.name_scope("test_tags"):
                    test_tags = get_all_tags(date1=test_data_params_["date1"],
                                             suffix=test_data_params_.get('suffix', ''))

                    def test_input_fn():
                        return self.input_fn(test_tags, data_params=test_data_params_)
            else:
                all_test_data_params = []
                with tf.name_scope("test_tags"):
                    test_tags = get_all_tags(date1=test_data_params_["date1"],
                                             suffix=test_data_params_.get('suffix', ''))

                    def get_input_dataset(data_params):
                        return self.input_fn(test_tags, data_params=data_params)

                    for campaign_spec in eval_campaign_specs:
                        test_data_params_ = test_data_params_.copy()
                        test_data_params_['campaign_initializer'] = CampaignInitializer.build_deterministic_from_spec(
                            campaign_spec, absolute=True, seed=tf_random_seed)
                        all_test_data_params.append(test_data_params_)

                    def test_input_fn():
                        print("Calling concatenated datasets input_fn")
                        dataset = get_input_dataset(all_test_data_params[0])
                        for test_param in all_test_data_params[1:]:
                            dataset = dataset.concatenate(get_input_dataset(test_param))
                        return dataset

            eval_spec = tf.estimator.EvalSpec(
                test_input_fn,
                steps=None,
                name='test',
                start_delay_secs=60,
                throttle_secs=60
            )

        eval_result, export_res = tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
        print("eval_result:", eval_result)
        return eval_result

    def predict(self, run_name, volume_target, missed_imp_penalty, test_date, predict_keys,
                checkpoint_path=None, common_data_params=None,
                run_config=None, warm_start_from=None, tf_random_seed=TF_RANDOM_SEED):
        predict_dirs = self.initialize_dir(run_name, False)
        estimator = self.estimator(model_dir=predict_dirs.checkpoint_dir, run_config=run_config,
                                   warm_start_from=warm_start_from, tf_random_seed=tf_random_seed,
                                   volume_target=volume_target, missed_imp_penalty=missed_imp_penalty)
        pred_results = {}
        eval_name = f"{test_date}_{volume_target}_{missed_imp_penalty}"
        print(f"Predict for {eval_name}...")
        fixed_data_params = {"repeat_count": 1, "date1": test_date}
        test_data_params = {}
        if common_data_params is not None:
            test_data_params.update(common_data_params)
        test_data_params.update(fixed_data_params)
        test_data_params["batch_size"] = test_data_params.get("batch_size", 100)
        test_data_params["grid_size_bid"] = test_data_params.get("grid_size_bid", GRID_SIZE_BID_EXP)
        test_data_params["suffix"] = test_data_params.get("suffix", '')
        test_data_params["cache_directory_root"] = self.cache_dir

        key = self.get_run_key(
            run_name=run_name, date=test_date, volume_target=volume_target, missed_imp_penalty=missed_imp_penalty)
        with tf.name_scope(f"test_tags_{test_date}"):
            test_tags = get_all_tags(date1=test_data_params['date1'], suffix=test_data_params.get('suffix', ''))
            pred_results[key] = estimator.predict(
                input_fn=lambda: self.input_fn(test_tags, data_params=test_data_params),
                predict_keys=predict_keys, yield_single_examples=False,
                checkpoint_path=checkpoint_path)
        return pred_results

    def eval(self, run_name, volume_target, missed_imp_penalty, test_dates, checkpoint_path=None,
             common_data_params=None, run_config=None, warm_start_from=None, tf_random_seed=TF_RANDOM_SEED,
             use_cache=True):
        eval_dirs = self.initialize_dir(run_name, False)
        estimator = self.estimator(model_dir=eval_dirs.eval_dir, run_config=run_config,
                                   warm_start_from=warm_start_from, tf_random_seed=tf_random_seed,
                                   volume_target=volume_target, missed_imp_penalty=missed_imp_penalty)

        eval_results = {}
        for test_date in test_dates:
            eval_name = f"{test_date}_{volume_target}_{missed_imp_penalty}"
            result = {}
            key = self.get_run_key(
                run_name=run_name, date=test_date, volume_target=volume_target, missed_imp_penalty=missed_imp_penalty)
            if use_cache and self.is_in_cache(run_name=run_name, eval_name=eval_name):
                print(f"Using cache for {eval_name}...")
                result = self.get_cache_eval(run_name=run_name, eval_name=eval_name)
            if len(result) == 0:
                print(f"Evaluating {eval_name} ...")
                fixed_data_params = {"repeat_count": 1, "date1": test_date}
                test_data_params = {}
                if common_data_params is not None:
                    test_data_params.update(common_data_params)
                test_data_params.update(fixed_data_params)
                test_data_params["batch_size"] = test_data_params.get("batch_size", 100)
                test_data_params["grid_size_bid"] = test_data_params.get("grid_size_bid", GRID_SIZE_BID_EXP)
                test_data_params["suffix"] = test_data_params.get("suffix", '')
                test_data_params["cache_directory_root"] = test_data_params.get("cache_directory_root",
                                                                                self.cache_dir)

                with tf.name_scope(f"test_tags_{test_date}"):
                    test_tags = get_all_tags(date1=test_data_params['date1'], suffix=test_data_params.get("suffix", ""))
                    try:
                        result = estimator.evaluate(
                            lambda: self.input_fn(test_tags, data_params=test_data_params),
                            checkpoint_path=checkpoint_path, name=eval_name,
                            hooks=self.get_hooks(estimator, dirs=eval_dirs, eval_name=f"eval_{eval_name}"))
                    except tf.errors.OpError as e:
                        eval_results[key] = str(e)
            if len(result) > 0:
                eval_results[key] = result
        return eval_results

    def is_in_cache(self, run_name, eval_name):
        return len(self._list_of_events(run_name=run_name, eval_name=eval_name)) > 0

    def get_cache_eval(self, run_name, eval_name):
        return parse_eval_events(self.get_latest_event(run_name, eval_name))

    def get_latest_event(self, run_name, eval_name):
        list_of_events = self._list_of_events(run_name=run_name, eval_name=eval_name)
        if len(list_of_events) > 0:
            return max(list_of_events, key=os.path.getctime)
        else:
            return None

    def _list_of_events(self, run_name, eval_name):
        event_dirs = self.initialize_dir(run_name, False)
        print(f"Looking for events in {os.path.join(event_dirs.eval_dir, f'eval_{eval_name}', 'events.out.tfevents.*')}")
        return glob.glob(os.path.join(event_dirs.eval_dir, f"eval_{eval_name}", "events.out.tfevents.*"))

    @staticmethod
    def get_hooks(estimator, dirs, eval_name=''):
        hooks = []
        if "VENOM_BUCKET_NAME" in os.environ:
            eval_dir = os.path.normpath(os.path.join(dirs.run_dir, eval_name))
            if len(eval_name) == 0:
                copy_func = tools.cp_checkpoints
            else:
                copy_func = tools.cp_evaluations
            s3_checkpoints_saving_listener = tools.S3SavingListener(
                lambda: copy_func(eval_dir, os.environ["VENOM_BUCKET_NAME"]))

            hooks.append(tf.train.CheckpointSaverHook(
                estimator.model_dir,
                save_steps=estimator.config.save_checkpoints_steps,
                listeners=[s3_checkpoints_saving_listener]))
        return hooks


def _default_update(first_dict, name, update_dict):
    update_dict = update_dict.copy()
    update_dict.update(first_dict.get(name, {}))
    first_dict[name] = update_dict


def get_run_name_prefix(params):
    prefix = []
    if "clip_norm" in params:
        prefix.append("c{:g}".format(params["clip_norm"]))
    reg = params.get("regularization", 0)
    if reg > 0:
        prefix.append("r{:g}".format(reg))
    spread_reg = params.get("spread_reg", 0)
    if spread_reg > 0:
        prefix.append("s{:g}".format(spread_reg))
    return "_".join(prefix)+("/" if prefix else "")


def parse_run_name(run_name):
    params = {}
    sp = re.split(r'(\d+\.\d+|\d+)', run_name)[::-1]
    for name in ["clip_norm", "regularization", "spread_reg"]:
        if len(sp) > 1 and sp[-1].replace('_', '') == name[0]:
            sp.pop()
            params[name] = float(sp.pop())
    params["run_name"] = run_name.replace(get_run_name_prefix(params), '')
    return params


def parse(model_path):
    # gamma/constant_noise_stacked_gru2/sgd_0.1_200_0.5_T/c1/new

    def _parse_actuator_type(model_args, experiment_args):
        actuator_type = model_args.pop()
        if 'dirac' in actuator_type:
            actuator_type = actuator_type.replace('dirac_', 'dirac_wrapper_')
        params = experiment_args.get("params", {})
        params['actuator_type'] = actuator_type
        experiment_args["params"] = params

    def _parse_controller_type(model_args, experiment_args):
        if model_args[-1] == "rnn":
            params = experiment_args.get("params", {})
            params["controller_type"] = model_args.pop()
            experiment_args["params"] = params

    def _parse_control_type(model_args, experiment_args):
        experiment_args["control_type"] = model_args.pop()

    def _parse_extra_control_input_names(model_args, experiment_args):
        extra_map = {'c': 'cost', 'v': 'volume', 'vt': 'volume_target'}
        if all(s in extra_map.keys() for s in model_args[-1].split('_')):
            extra_control_input_n = model_args.pop().split('_')
            controller_params = experiment_args.get("controller_params", {})
            controller_params["extra_control_input_names"] = tuple(extra_map[e] for e in extra_control_input_n)
            experiment_args["controller_params"] = controller_params

    def _parse_relative(model_args, experiment_args):
        if model_args[-1] == "relative":
            model_args.pop()
            controller_params = experiment_args.get("controller_params", {})
            controller_params["relative"] = True
            experiment_args["controller_params"] = controller_params

    def _parse_ctrl_init(model_args, experiment_args):
        current_params = model_args.pop()
        # used only for pi control: we simplify this step
        # "dirac_gamma/rnn/pi/c/relative/ctrli_0.0_0.2_F__0_F/rms_0.001/BASE_regL1_0.01_L2_0.01"
        control_params = experiment_args.get('control_params', {})
        if current_params.startswith("ctrli"):
            assert experiment_args["control_type"].startswith('pi')
            ctrli = current_params.replace("ctrli_", "").split('__')
            init_noise = np.array([float(k) for k in ctrli[0][:-2].split('_')])
            if experiment_args["control_type"].startswith('pid'):
                control_params["initial_state_params"] = PIDControlCell.State(
                    control_output=(init_noise, ctrli[0][-1] == 'T'),
                    i_error=(float(ctrli[1][:-2]), ctrli[1][-1] == 'T'),
                    previous_error=(1, False)
                )
            else:
                control_params["initial_state_params"] = PIControlCell.State(
                    control_output=(init_noise, ctrli[0][-1] == 'T'), i_error=(float(ctrli[1][:-2]), ctrli[1][-1] == 'T'))
            experiment_args['control_params'] = control_params
        else:
            model_args.append(current_params)

    def _parse_pi_params(model_args, experiment_args):
        current_params = model_args.pop()
        control_params = experiment_args.get('control_params', {})
        # control_params["initial_state_params"]
        if current_params.startswith("PI_") or current_params.startswith("PID_"):
            # PI_k{k_l}_ti{t_i}_ts{t_s}_pg{plant_gain_var:e}_log{'T' if use_log else 'F'}")
            match_params = re.match(
                r"PID?_k(?P<k_l>[^_]+)_ti(?P<t_i>.*)_ts(?P<t_s>.*)_pg(?P<pg>[^_]*)_log(?P<use_log>\w)",
                current_params)
            control_params['k_l'] = float(match_params.group('k_l'))
            control_params['t_i'] = float(match_params.group('t_i'))
            control_params['t_s'] = float(match_params.group('t_s'))
            control_params['plant_gain_var'] = float(match_params.group('pg'))
            control_params['use_log'] = match_params.group('use_log') == 'T'
            experiment_args['control_params'] = control_params
        else:
            model_args.append(current_params)

    def _parse_optimizer(model_args, experiment_args):
        optim_param = model_args.pop()
        # parse optimizer
        if optim_param.startswith('rms'):
            sp = optim_param.split('_')
            go = GenericOptimizer(initializer=tf.train.RMSPropOptimizer, name='_'.join(sp[:-1]),
                                  learning_rate=float(sp[-1]))
        elif optim_param.startswith('adam'):
            sp = optim_param.split('_')
            go = GenericOptimizer(initializer=tf.train.AdamOptimizer, name='_'.join(sp[:-1]),
                                  learning_rate=float(sp[-1]))
        elif optim_param.startswith("sgd"):
            if "DK" in optim_param:
                if 'cosWRDK' in optim_param:
                    match_params = re.match(
                        (r"sgd_cosWRDK_(?P<ilr>\d+\.\d*)(_(?P<ds>\d+\.?\d*))?_(?P<t_mul>\d+\.\d*)"
                         r"_(?P<m_mul>\d+\.\d*)_(?P<alpha>\d+\.\d*)"),
                        optim_param)
                    initial_learning_rate = match_params.group('ilr')
                    decay_steps = match_params.group('ds')
                    if decay_steps is None:  # Quick fix for legacy runs to be reusable # TODO(at): remove
                        decay_steps = 50
                    t_mul = match_params.group('t_mul')
                    m_mul = match_params.group('m_mul')
                    alpha = match_params.group('alpha')
                    go = AnnealedSGDOptimizer(initial_learning_rate=initial_learning_rate,
                                              decay_steps=decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha)
                else:
                    match_params = re.match(
                        r"sgd_(?P<decay_fn>[^_]+)_(?P<ilr>\d+\.\d*)_(?P<ds>\d+\.?\d*)_(?P<dr>\d+\.\d*)_(?P<sc>\w)",
                        optim_param)
                    decay_fn = tf.train.exponential_decay if match_params.group(
                        'decay_fn') == "expDK" else tf.train.inverse_time_decay
                    initial_learning_rate = match_params.group('ilr')
                    decay_steps = match_params.group('ds')
                    decay_rate = match_params.group('dr')
                    staircase = match_params.group('sc')
                    go = ScheduledSGDOptimizer(initial_learning_rate=initial_learning_rate,
                                               decay_steps=decay_steps,
                                               decay_rate=decay_rate,
                                               staircase=staircase,
                                               decay_fn=decay_fn)
            else:
                sp = optim_param.split('_')
                go = GenericOptimizer(initializer=tf.train.GradientDescentOptimizer,
                                      name='_'.join(sp[:-1]), learning_rate=sp[-1])
        else:
            raise ValueError(f"{optim_param} not supported.")
        experiment_args['params'].update(go.to_model_params())

    def _parse_run_name(model_args, experiment_args):
        params = parse_run_name("/".join(model_args[::-1]))
        run_name = params.pop("run_name")
        experiment_args["params"].update(params)
        return run_name

    parsers = [_parse_actuator_type, _parse_controller_type, _parse_control_type, _parse_extra_control_input_names,
               _parse_relative, _parse_ctrl_init, _parse_pi_params, _parse_optimizer, _parse_run_name]

    model_args_ = model_path.split("/")[::-1]
    experiment_args_ = {}
    run_name_ = ''
    for parser in parsers:
        run_name_ = parser(model_args_, experiment_args_)  # last parser returns the `run_name`.
    return experiment_args_, run_name_
