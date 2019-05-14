from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from rnn_controller.constants import FLOAT_TYPE, BID_SCALE_EXP, GRID_SIZE_TIME
from rnn_controller.data import CampaignSpec
from rnn_controller.evaluation import TEST_DATES, Evaluation
from rnn_controller.toy_volume_curves import ToyVolumeCurves


def jitter_stripplot(df, x="mean_stopped_cost_and_penalty"):
    # https://seaborn.pydata.org/examples/jitter_stripplot.html#jitter-stripplot
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)
    sns.stripplot(x=x, y="control_type", hue="volume_target",
                  data=df, dodge=True, jitter=True,
                  alpha=.25, zorder=1)
    sns.pointplot(x=x, y="control_type", hue="volume_target",
                  data=df, dodge=.532, join=False, palette="dark",
                  markers="d", scale=.75, ci=None)
    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:], labels[3:], title="volume_target",
              handletextpad=0, columnspacing=1,
              loc="lower left", ncol=3, frameon=True)
    plt.show()


def cat_plot(df, y, y_label):
    g = sns.catplot(x="volume_target", y=y, hue="control_type", data=df,
                    height=6, kind="bar", palette="muted", legend=False)
    g.despine(left=True)
    g.set_ylabels(y_label)
    plt.legend(title="control_type", loc="lower right", frameon=True)
    plt.show()


def plot_ratios_for_model(df, model_name):
    # plot_ratios_for_model(df[df.control_type == "constant_noise_stacked_gru16"])
    df.plot(x="volume_target", y="cost_penalty_ratio", label=model_name)
    plt.ylabel("ratio with respect to the benchmark")
    plt.show()


def plot_cost(df):
    # plot_ratios_for_model(df[df.control_type == "constant_noise_stacked_gru16"])
    df = df.setindex('volume_target')
    df.groupby('control_type')['mean_stopped_cost_and_penalty'].plot(legend=True)
    plt.ylabel("percentage of the max cost")
    plt.show()


def plot_cost_and_penalty_distributions(evaluations, campaign_specs: CampaignSpec, volume_pattern_provider_fn=None,
                                        test_dates=None, common_data_params=None):
    if test_dates is None:
        test_dates = TEST_DATES

    # predict_keys = ['total_realized_volume', 'total_cost', 'total_cost_stopped',
    #                 'total_penalty', 'total_stopped_cost_and_penalty', ]
    predict_keys = ['total_cost_stopped_and_penalty', ]

    results = {}
    for evaluation in evaluations:
        experiment, run_name = evaluation.build_experiment(volume_pattern_provider_fn=volume_pattern_provider_fn)

        [results.update(experiment.predict(run_name, test_date=test_date,
                                           predict_keys=predict_keys,
                                           common_data_params=common_data_params,
                                           checkpoint_path=evaluation.checkpoint_path,
                                           volume_target=campaign_spec.volume_target,
                                           missed_imp_penalty=campaign_spec.missed_imp_penalty)
                        ) for campaign_spec in campaign_specs for test_date in test_dates]

    def flatten_results(pred_iter, keys):
        results = {key: [] for key in keys}
        for pred in pred_iter:
            for key in keys:
                results[key].append(pred[key])
        return {key: np.concatenate(results[key]) for key in keys}

    def merge_dates(results):
        merged_results = defaultdict(lambda: defaultdict(lambda: np.empty(shape=(1, 1))))
        for run_key in sorted(results.keys(), key=lambda k: k.date):
            run_key_all_date = run_key._replace(date='ALL')
            for key in results[run_key].keys():
                merged_results[run_key_all_date][key] = np.concatenate((merged_results[run_key_all_date][key],
                                                                        results[run_key][key]))
        return merged_results

    def plot_distrib(fig, ax, run_key, run_results, label=None, color=None):
        # costs = run_results['total_cost']
        # volumes = run_results['total_realized_volume']
        # losses = costs + np.maximum(0, run_key.volume_target - volumes) * run_key.missed_imp_penalty
        # print(losses-(run_results['total_cost']+run_results['total_penalty']))
        losses = run_results['total_cost_stopped_and_penalty']

        def plot_fn(x, ax=ax):
            bins = np.linspace(0, 1.2*run_key.volume_target*run_key.missed_imp_penalty, 60)
            hist_kws = {'density': True}
            if label is not None:
                hist_kws['label'] = label
            if color is not None:
                hist_kws['color'] = color
            ax = sns.distplot(x, bins=bins, kde=False, hist_kws=hist_kws, ax=ax)
            ax.grid(True)
            return ax

        plot_fn(losses, ax=ax)
        ax.axvline(x=np.mean(losses), linewidth=2, label='mean ' + (label if label is not None else ''),
                   color='r' if color is None else color, linestyle=':')
        ax.legend(loc='upper right')
        print(np.mean(losses))
        ax.set_xlabel('campaign cost')
        ax.set_ylabel('count')

    def plot_distrib_cdf(fig, ax, run_key, run_results, label=None, color=None):
        losses = run_results['total_cost_stopped_and_penalty']

        def plot_fn(x, ax=ax):
            bins = np.linspace(0, 1.2*run_key.volume_target*run_key.missed_imp_penalty, 60)
            hist_kws = {'density': True, 'histtype': 'step', 'cumulative': True, 'linewidth': 2}
            if label is not None:
                hist_kws['label'] = label
            if color is not None:
                hist_kws['color'] = color
            ax = sns.distplot(x, bins=bins, kde=False, hist_kws=hist_kws, ax=ax)
            ax.grid(True)
            return ax

        plot_fn(losses, ax=ax)
        ax.axvline(x=np.mean(losses), linewidth=2, label='mean ' + (label if label is not None else ''),
                   color='r' if color is None else color, linestyle=':')
        ax.legend(loc='lower right')
        print(np.mean(losses))
        ax.set_xlabel('campaign cost')

    def plot_distrib_difference(fig, ax, run_key, run_results, run_key2, run_results2,
                                label=None, color=None):
        losses_difference = run_results['total_cost_stopped_and_penalty'] \
                            - run_results2['total_cost_stopped_and_penalty']

        def plot_fn(x, ax=ax):
            bins = np.linspace(-run_key.volume_target*run_key.missed_imp_penalty,
                               run_key.volume_target*run_key.missed_imp_penalty, 51)
            hist_kws = {'density': True}
            if label is not None:
                hist_kws['label'] = label
            if color is not None:
                hist_kws['color'] = color
            ax = sns.distplot(x, bins=bins, kde=False, hist_kws=hist_kws, ax=ax)
            ax.grid(True)
            return ax

        plot_fn(losses_difference, ax=ax)
        ax.axvline(x=np.mean(losses_difference), linewidth=2, label='mean ' + (label if label is not None else ''),
                   color='r' if color is None else color, linestyle=':')
        ax.legend(loc='upper right')
        print(np.mean(losses_difference))
        ax.set_xlabel('campaign cost difference')
        ax.set_ylabel('density')

    def plot_distrib_difference_cdf(fig, ax, run_key, run_results, run_key2, run_results2,
                                    label=None, color=None):
        losses_difference = run_results['total_cost_stopped_and_penalty'] \
                            - run_results2['total_cost_stopped_and_penalty']

        def plot_fn(x, ax=ax):
            bins = np.linspace(-run_key.volume_target*run_key.missed_imp_penalty,
                               run_key.volume_target*run_key.missed_imp_penalty, 51)
            hist_kws = {'density': True, 'histtype': 'step', 'cumulative': True, 'linewidth': 2}
            if label is not None:
                hist_kws['label'] = label
            if color is not None:
                hist_kws['color'] = color
            ax = sns.distplot(x, bins=bins, kde=False, hist_kws=hist_kws, ax=ax)
            ax.grid(True)
            return ax

        plot_fn(losses_difference, ax=ax)
        ax.axvline(x=np.mean(losses_difference), linewidth=2, label='mean ' + (label if label is not None else ''),
                   color='r' if color is None else color, linestyle=':')
        ax.legend(loc='lower right')
        print(np.mean(losses_difference))
        ax.set_xlabel('campaign cost difference')
        ax.set_ylabel('CDF')

    results = {k: flatten_results(v, predict_keys) for k, v in results.items()}
    results = merge_dates(results)
    print(results.keys())
    color_palette = sns.color_palette()
    colors = [color_palette[1], color_palette[4]]
    fig, ax = plt.subplots(len(results), 1, sharex=True, sharey=True, figsize=(12, 10))
    for e, run_key in enumerate(results):
        plot_distrib(fig, ax[e], run_key, results[run_key], color=colors[e],
                     label='PI' if 'benchmark' in run_key.run_name else 'RNN')
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(12, 10))
    PI_run_key = None
    RNN_run_key = None
    for e, run_key in enumerate(results):
        # plot_distrib_cdf(fig, ax[0], run_key, results[run_key], color=colors[e],
        #                  label='PI' if 'benchmark' in run_key.run_name else 'RNN')
        if 'benchmark' in run_key.run_name:
            PI_run_key = run_key
        else:
            RNN_run_key = run_key
    plot_distrib_difference_cdf(fig, ax[0], PI_run_key, results[PI_run_key],
                                RNN_run_key, results[RNN_run_key], color=colors[1],
                                label='PI cost - RNN cost')
    plot_distrib_difference(fig, ax[1], PI_run_key, results[PI_run_key],
                            RNN_run_key, results[RNN_run_key],
                            color=colors[1], label='PI cost - RNN cost')
    plt.show()


def volume_shock_plots_multiple(evaluations, campaign_spec: CampaignSpec, daily_volume, normalized_volume=False,
                                base_volume_curve=None,
                                change_times=(96, 192),
                                factors=(1.5, 2, 5, 10, 20, 50),
                                factor2multiplier_fn=lambda x: [1. / x, x],
                                volume_pattern_provider_fn=None, run_config=None,
                                bid_scale=BID_SCALE_EXP,
                                num_timesteps=GRID_SIZE_TIME, dtype=FLOAT_TYPE,
                                keys_to_plot=['bid_level', 'noise_level', 'volume', 'cost']):
    print(np.sum(base_volume_curve[..., -1]))
    if daily_volume is not None and len(base_volume_curve.shape) > 1:
        daily_norm_factor = daily_volume/np.sum(base_volume_curve[..., -1])
        base_volume_curve = base_volume_curve * daily_norm_factor
    print(np.sum(base_volume_curve[..., -1]))
    if base_volume_curve is None:
        daily_norm_factor = daily_volume / (ToyVolumeCurves.BASE[-1] * num_timesteps)
        base_volume_curve = ToyVolumeCurves.BASE * daily_norm_factor
    toy_curves = ToyVolumeCurves(daily_volume if normalized_volume else None,
                                 num_timesteps, base_volume_curve=base_volume_curve, bid_scale=bid_scale)
    constant_curves = toy_curves.constant_volume()

    volume_shocked_curves = [constant_curves, ] + [
        toy_curves.volume_shocks(change_times=change_times,
                                 multipliers=factor2multiplier_fn(x)) for x in factors]
    # print(volume_shocked_curves)
    def toy_dataset_fn(curves):

        return tf.data.Dataset.from_tensor_slices(np.stack(curves)).map(lambda p: tf.cast(p, dtype))

    def get_predictions(evaluation: Evaluation, curves):
        vp_fn = volume_pattern_provider_fn if 'benchmark' in evaluation.run_name else None
        experiment, run_name = evaluation.build_experiment(volume_pattern_provider_fn=vp_fn,
                                                           nb_timestamps=num_timesteps,
                                                           bid_scale=bid_scale)
        dirs = experiment.initialize_dir(run_name, restart=False)
        estimator = experiment.estimator(model_dir=dirs.checkpoint_dir, run_config=run_config,
                                         volume_target=campaign_spec.volume_target,
                                         missed_imp_penalty=campaign_spec.missed_imp_penalty)
        predict_keys = ["cost", "control_output", "volume", "total_stopped_cost", "total_penalty"]
        return next(estimator.predict(input_fn=lambda: toy_dataset_fn(curves).batch(len(curves)),
                                      predict_keys=predict_keys,
                                      yield_single_examples=False,
                                      checkpoint_path=evaluation.checkpoint_path))

    def fill_nan_horz2d(arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
        return arr

    def fill_nan(arr):
        for i in range(arr.shape[0]):
            fill_nan_horz2d(arr[i].T)
        return arr

    def stopped_predictions(predict_res):
        # print(predict_res['cost'].shape)
        # print(predict_res)
        volume_cum = predict_res['volume']
        stop_mask = volume_cum > campaign_spec.volume_target
        volume_cum[stop_mask] = np.nan
        predict_res['volume'] = fill_nan(volume_cum)
        cost = predict_res['cost']
        cost[stop_mask] = np.nan
        predict_res['cost'] = fill_nan(cost)
        control_output = predict_res['control_output']
        bid_level = control_output[..., 0]
        noise_level = control_output[..., 1]
        bid_level[stop_mask[..., 0]] = np.nan
        noise_level[stop_mask[..., 0]] = np.nan
        control_output = np.stack([bid_level, noise_level], axis=-1)
        predict_res['control_output'] = control_output
        return predict_res

    def _plot(ax, value, ylabel, colors):
        # ax.plot(value.T)
        for ix in range(value.shape[0]):
            ax.plot(value[ix, :].T, color=colors[ix])
        nice_label = {'bid_level': 'bid level', 'volume': 'volume', 'cost': 'spend'}
        ax.set_ylabel(nice_label[ylabel])

    def plot(fig, axs, evaluation, predict_res, title, colors, keys_to_plot=None):
        # Remove horizontal space between axes
        # fig.subplots_adjust(hspace=0.) # This works only when saved as pdf
        if title is not None:
            plt.suptitle(title, y=1.)
        i = 0
        key2axis = {k: e for e, k in enumerate(keys_to_plot)} if keys_to_plot is not None else None
        for key, value in predict_res.items():
            value = np.squeeze(value)
            if key == 'control_output':
                key = 'bid_level'
                if keys_to_plot is None or key in keys_to_plot:
                    idx = i if key2axis is None else key2axis[key]
                    _plot(axs[idx], value[..., 0], key, colors=colors)
                    i += 1
                value = value[..., 1]
                key = 'noise_level'
                if 'benchmark' in evaluation.run_name:
                    value = np.zeros_like(value)
            if keys_to_plot is not None and key not in keys_to_plot:
                print(key)
                continue
            idx = i if key2axis is None else key2axis[key]
            _plot(axs[idx], value, key, colors=colors)
            i += 1
        axs[-1].set_xlabel('time')
        #plt.tight_layout()
        # axs[0].legend(labels=[f'{f:g}' for f in [1] + factors], fancybox=True, framealpha=0.5, loc='upper left')

    def penalty_plot(fig, ax, evaluation, predict_res, colors):
        fig.subplots_adjust(hspace=0.)
        cost = np.squeeze(predict_res['total_stopped_cost'])
        penalty = np.squeeze(predict_res['total_penalty'])
        ind = np.arange(len(cost))
        ax.bar(ind, cost, tick_label=[f'{f:g}' for f in [1] + factors], color=colors[0])
        ax.bar(ind, penalty, bottom=cost, color=colors[1])
        ax.set_ylabel("final cost (spend + penalty)")
        ax.set_xlabel("shock factor")

    title_suffix = f'target:{campaign_spec.volume_target}, penalty:{campaign_spec.missed_imp_penalty}' \
                   + f' - Base Total Volume: {daily_volume}{" - Renormalized "if normalized_volume else ""}'

    # colorbrewer qualitative paired palette (with one color changed)
    # colornames = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
    #               '#6a3d9a', '#ffff99', '#b15928', ]
    colornames = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
                  '#6a3d9a', '#d8ac93', '#b15928', ]

    color_palette_dark = sns.color_palette(colornames[1::2])
    color_palette_light = sns.color_palette(colornames[::2])
    # fig, axs = plt.subplots(len(keys_to_plot), len(evaluations), num=1, sharex=True, sharey='row', figsize=(12, 10))
    # fig1, axs1 = plt.subplots(1, len(evaluations), num=2, sharex=True, sharey='row', figsize=(12, 10))
    fig, axs = plt.subplots(len(keys_to_plot)+1, len(evaluations), num=1, sharex=False, sharey='row', figsize=(12, 10))
    # shared_axes = axs[0, 0].get_shared_x_axes()
    # print(shared_axes)
    # for i in range(len(evaluations)):
    #     shared_axes.remove(axs[-1, i])
    for e, evaluation_ in enumerate(evaluations):
        predict_res = get_predictions(evaluation_, volume_shocked_curves)
        predict_res = stopped_predictions(predict_res)
        # plot(fig, axs[:, e], evaluation_, predict_res, colors=color_palette_dark, title=None, keys_to_plot=keys_to_plot)
        # penalty_plot(fig1, axs1[e], evaluation_, predict_res, colors=[color_palette_dark, color_palette_light])
        plot(fig, axs[:-1, e], evaluation_, predict_res, colors=color_palette_dark, title=None, keys_to_plot=keys_to_plot)
        penalty_plot(fig, axs[-1, e], evaluation_, predict_res, colors=[color_palette_dark, color_palette_light])

    plt.figure(1)
    plt.savefig("/tmp/volume_shock_plots_multiple_fig1.pdf")
    plt.show()

    def plot_volume_profile(curves):
        with sns.color_palette(color_palette_dark):
            for curve in curves:
                plt.plot(np.cumsum(curve[:, -1]))
            plt.show()

    plot_volume_profile(volume_shocked_curves)


def plot_toy_data_learn_vs_real(evaluations, campaign_spec: CampaignSpec, input_fns=None,
                                volume_pattern_provider_fn=None, run_config=None,
                                bid_scale=BID_SCALE_EXP,
                                num_timesteps=GRID_SIZE_TIME, dtype=FLOAT_TYPE):


    def get_predictions(evaluation: Evaluation, input_fn, keys=None):
        #dict_keys(['response', 'volume', 'cost', 'time', 'control_output', 'realized_volume_cum', 'total_realized_volume', 'total_cost', 'total_stopped_cost', 'total_penalty', 'total_stopped_cost_and_penalty'])
        vp_fn = volume_pattern_provider_fn if 'benchmark' in evaluation.run_name else None
        experiment, run_name = evaluation.build_experiment(volume_pattern_provider_fn=vp_fn,
                                                           nb_timestamps=num_timesteps,
                                                           bid_scale=bid_scale)
        dirs = experiment.initialize_dir(run_name, restart=False)
        estimator = experiment.estimator(model_dir=dirs.checkpoint_dir, run_config=run_config,
                                         volume_target=campaign_spec.volume_target,
                                         missed_imp_penalty=campaign_spec.missed_imp_penalty)
        return next(estimator.predict(input_fn=input_fn,
                                      predict_keys=keys,
                                      yield_single_examples=False,
                                      checkpoint_path=evaluation.checkpoint_path))

    def fill_nan_horz2d(arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
        return arr

    def fill_nan(arr):
        for i in range(arr.shape[0]):
            fill_nan_horz2d(arr[i].T)
        return arr

    def stopped_predictions(predict_res):
        # print(predict_res['cost'].shape)
        # print(predict_res.keys())
        volume_cum = predict_res['volume']
        stop_mask = volume_cum > campaign_spec.volume_target
        volume_cum[stop_mask] = np.nan
        predict_res['volume'] = fill_nan(volume_cum)
        cost = predict_res['cost']
        cost[stop_mask] = np.nan
        predict_res['cost'] = fill_nan(cost)
        # control_output = predict_res['control_output']
        # bid_level = control_output[..., 0]
        # noise_level = control_output[..., 1]
        # bid_level[stop_mask[..., 0]] = np.nan
        # noise_level[stop_mask[..., 0]] = np.nan
        # control_output = np.stack([bid_level, noise_level], axis=-1)
        # predict_res['control_output'] = control_output
        return predict_res

    def _plot(ax, value, ylabel):
        ax.plot(value.T)
        ax.set_ylabel(ylabel)

    def _dist(ax, value, xlabel, bins):
        # hist_kws = {'density': True, 'histtype': 'step', 'cumulative': True, 'linewidth': 2}
        # ax = sns.distplot(value.T, bins=bins, kde=False, hist_kws=hist_kws, ax=ax)
        ax.hist(value.T, bins=bins, weights=np.ones((len(value),))/len(value), color='#ff7f00')
        ax.hist(value.T, bins=bins, density=True, histtype='step', cumulative=True, linewidth=2, color='#6a3d9a')
        ax.axvline(x=np.mean(value.T), linewidth=3, color='#e31a1c', linestyle=':')
        # ax.legend(loc='upper right')
        key2label = {'total_realized_volume': 'final volume',
                     'total_stopped_cost_and_penalty': 'final cost (spend + penalty)'}
        ax.set_xlabel(key2label[xlabel])
        ax.set_ylabel('proportion')
        ax.grid()

    def plot(fig, ax, predict_res, key, timeplot=False, bins=None):
        # Remove horizontal space between axes
        fig.subplots_adjust(hspace=0.)
        value = np.squeeze(predict_res[key])
        if timeplot:
            _plot(ax, value, key)
        else: #plot final state dist
            _dist(ax, value, key, bins=bins)
        plt.tight_layout()

    def stats(predict_res):
        cost = predict_res['total_stopped_cost_and_penalty']
        volume = predict_res['total_realized_volume']
        n = len(volume)
        shortfall_prob = np.count_nonzero(volume < campaign_spec.volume_target) / n
        return np.mean(cost), np.std(cost), shortfall_prob, np.mean(volume), np.percentile(volume, 50),\
               np.count_nonzero(volume < 0.95*campaign_spec.volume_target) / n,\
               np.count_nonzero(volume < 0.98*campaign_spec.volume_target) / n,\
               np.count_nonzero(volume < 0.99*campaign_spec.volume_target) / n

    keys = ['volume', 'cost', 'total_realized_volume', 'total_stopped_cost', 'total_penalty', 'total_stopped_cost_and_penalty']

    fig1, axs1 = plt.subplots(len(evaluations), len(input_fns), num=1, sharex=True, sharey=True, figsize=(12, 10))
    fig2, axs2 = plt.subplots(len(evaluations), len(input_fns), num=2, sharex=True, sharey=True, figsize=(12, 10))
    fig3, axs3 = plt.subplots(len(evaluations), len(input_fns), num=3, sharex=True, sharey=True, figsize=(12, 10))
    for e, evaluation_ in enumerate(evaluations):
        for e1, input_fn in enumerate(input_fns):
            predict_res = get_predictions(evaluation_, input_fn, keys=keys)
            predict_res = stopped_predictions(predict_res)
            # plot(fig, axs[e, e1], predict_res, 'cost', timeplot=True)
            # 'total_stopped_cost', 'total_penalty', 'total_stopped_cost_and_penalty'
            plot(fig1, axs1[e, e1], predict_res, 'total_stopped_cost_and_penalty', timeplot=False,
                 bins=np.linspace(0, 2000, 50))
            plot(fig2, axs2[e, e1], predict_res, 'total_realized_volume', timeplot=False,
                 bins=np.linspace(0, 200, 50))
            plot(fig3, axs3[e, e1], predict_res, 'volume', timeplot=True)
            print("STATS:", evaluation_, e1, stats(predict_res))

    plt.figure(1)
    plt.savefig("/tmp/plot_toy_data_learn_vs_real_fig1.pdf")
    plt.figure(2)
    plt.savefig("/tmp/plot_toy_data_learn_vs_real_fig2.pdf")
    plt.show()


def volume_shock_plots(evaluation, campaign_spec: CampaignSpec, daily_volume=None, normalized_volume=False,
                       base_volume_curve=None,
                       change_times=(96, 192),
                       factors=(1.5, 2, 5, 10, 20, 50),
                       factor2multiplier_fn=lambda x: [1./x, x],
                       volume_pattern_provider_fn=None, run_config=None,
                       num_timesteps=GRID_SIZE_TIME, dtype=FLOAT_TYPE):
    print(np.sum(base_volume_curve[..., -1]))
    if daily_volume is not None and len(base_volume_curve.shape) > 1:
        daily_norm_factor = daily_volume/np.sum(base_volume_curve[..., -1])
        base_volume_curve = base_volume_curve / daily_norm_factor
    print(np.sum(base_volume_curve[..., -1]))
    if base_volume_curve is None:
        daily_norm_factor = daily_volume/(ToyVolumeCurves.BASE[-1]*num_timesteps)
        base_volume_curve = ToyVolumeCurves.BASE * daily_norm_factor
    toy_curves = ToyVolumeCurves(daily_volume if normalized_volume else None,
                                 num_timesteps, base_volume_curve=base_volume_curve)
    constant_curves = toy_curves.constant_volume()

    volume_shocked_curves = [constant_curves, ] + [
        toy_curves.volume_shocks(change_times=change_times,
                                 multipliers=factor2multiplier_fn(x)) for x in factors]

    def toy_dataset_fn(curves):
        return tf.data.Dataset.from_tensor_slices(np.stack(curves)).map(lambda p: tf.cast(p, dtype))

    experiment, run_name = evaluation.build_experiment(volume_pattern_provider_fn=volume_pattern_provider_fn)
    experiment.initialize_dir(run_name, restart=False)
    estimator = experiment.estimator(model_dir=experiment.checkpoint_dir, run_config=run_config,
                                     volume_target=campaign_spec.volume_target,
                                     missed_imp_penalty=campaign_spec.missed_imp_penalty)

    def _plot(ax, value, ylabel):
        ax.plot(value.T)
        ax.set_ylabel(ylabel)

    def plot(fig, axs, curves, title):
        keys_to_plot = ["cost", "control_output", "volume"]
        predict_res = next(estimator.predict(input_fn=lambda: toy_dataset_fn(curves).batch(len(curves)),
                                             predict_keys=keys_to_plot,
                                             yield_single_examples=False,
                                             checkpoint_path=evaluation.checkpoint_path))
        # Remove horizontal space between axes
        fig.subplots_adjust(hspace=0.)
        if title is not None:
            plt.suptitle(title, y=1.)
        i = 0
        for key, value in predict_res.items():
            value = np.squeeze(value)
            if key == 'control_output':
                _plot(axs[i], value[..., 0], 'bid_level')
                i += 1
                value = value[..., 1]
                key = 'noise_level'
            _plot(axs[i], value, key)
            i += 1
        axs[-1].set_xlabel('time')
        plt.tight_layout()
        axs[0].legend(labels=[f'{f:g}' for f in [1]+factors], fancybox=True, framealpha=0.5, loc='upper left')

    title_suffix = f'target:{campaign_spec.volume_target}, penalty:{campaign_spec.missed_imp_penalty}' \
                   + f' - Base Total Volume: {daily_volume}{" - Renormalized "if normalized_volume else ""}'

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    plot(fig, axs, volume_shocked_curves, f'Volume shocks - {title_suffix}\n{evaluation.run_name}')
    plt.show()
    # TODO(at) add volume pattern and bid profile visuals


def toy_curve_plots(evaluation: Evaluation, campaign_spec: CampaignSpec, daily_volume, normalized_volume=False,
                    change_times=(96, 192),
                    factors=(1.5, 2, 5, 10, 20, 50),
                    factor2multiplier_fn=lambda x: [1./x, x],
                    volume_pattern_provider_fn=None, run_config=None,
                    num_timesteps=GRID_SIZE_TIME, dtype=FLOAT_TYPE):
    daily_norm_factor = daily_volume/(ToyVolumeCurves.BASE[-1]*num_timesteps)
    toy_curves = ToyVolumeCurves(daily_volume if normalized_volume else None,
                                 num_timesteps, base_volume_curve=ToyVolumeCurves.BASE*daily_norm_factor)
    constant_curves = toy_curves.constant_volume()
    shifted_bid_curves = [constant_curves, ] + [
        toy_curves.shifted_bid_levels(change_times=change_times,
                                      multipliers=factor2multiplier_fn(x)) for x in factors]
    volume_shocked_curves = [constant_curves, ] + [
        toy_curves.volume_shocks(change_times=change_times,
                                 multipliers=factor2multiplier_fn(x)) for x in factors]

    def toy_dataset_fn(curves):
        return tf.data.Dataset.from_tensor_slices(np.stack(curves)).map(lambda p: tf.cast(p, dtype))

    experiment, run_name = evaluation.build_experiment(volume_pattern_provider_fn=volume_pattern_provider_fn)
    experiment_dir = experiment.initialize_dir(run_name, restart=False)
    estimator = experiment.estimator(model_dir=experiment_dir.checkpoint_dir, run_config=run_config,
                                     volume_target=campaign_spec.volume_target,
                                     missed_imp_penalty=campaign_spec.missed_imp_penalty)

    def _plot(ax, value, ylabel):
        ax.plot(value.T)
        ax.set_ylabel(ylabel)

    def plot(fig, axs, curves, title):
        keys_to_plot = ["cost", "control_output", "realized_volume_cum"]
        predict_res = next(estimator.predict(input_fn=lambda: toy_dataset_fn(curves).batch(len(curves)),
                                             predict_keys=keys_to_plot,
                                             yield_single_examples=False,
                                             checkpoint_path=evaluation.checkpoint_path))
        # Remove horizontal space between axes
        fig.subplots_adjust(hspace=0.)
        if title is not None:
            plt.suptitle(title, y=1.)
        i = 0
        for key, value in predict_res.items():
            value = np.squeeze(value)
            if key == 'control_output':
                _plot(axs[i], value[..., 0], 'bid_level')
                i += 1
                value = value[..., 1]
                key = 'noise_level'
            _plot(axs[i], value, key)
            i += 1
        axs[-1].set_xlabel('time')
        plt.tight_layout()
        axs[0].legend(labels=[f'{f:g}' for f in [1]+factors], fancybox=True, framealpha=0.5, loc='upper left')

    title_suffix = f'target:{campaign_spec.volume_target}, penalty:{campaign_spec.missed_imp_penalty}' \
                   + f' - Base Total Volume: {daily_volume}{" - Renormalized "if normalized_volume else ""}'

    fig, axs = plt.subplots(4, 2, sharex=True, figsize=(12, 10))
    plot(fig, axs[:, 0], shifted_bid_curves, None) #f'bid shocks - {title_suffix}'
    plot(fig, axs[:, 1], volume_shocked_curves, f'Bid (left) and Volume (right) shocks - {title_suffix}\n{evaluation.run_name}')
    plt.show()
    # TODO(at) add volume pattern and bid profile visuals
