from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import io
import logging
import os
import shutil
from collections import namedtuple

import boto3
import imageio
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow import GraphKeys


IMAGE_SUMMARIES_KEY = 'image_summaries'


def is_venom():
    return "VENOM" in os.environ


def clear_checkpoints(pth):
    """ Clear checkpoints folder """
    pth = getattr(pth, "checkpoint_dir", pth)  # Accepts FLAGS
    print("Clearing", pth)
    if os.path.exists(pth):
        shutil.rmtree(pth)


ExperimentDir = namedtuple("ExperimentDir", ["run_dir", "checkpoint_dir", "eval_dir", "plot_dir"])


class DirInitializer(object):
    """ Initialize checkpoint folder.
    - restart=False: start from existing checkpoint
    - restart=True: clear checkpoint
    - restart="string": start from another existing checkpoint

    """
    CACHE_DIR = '/tmp/tensorflow_cache/'
    CHECKPOINT_DIR_ROOT = '/tmp/tensorflow/'
    EVAL_ROOT = '/tmp/tensorflow_eval/'
    PLOT_ROOT = '/tmp/plots/'

    def __init__(self, cache_dir=CACHE_DIR, checkpoint_dir_root=CHECKPOINT_DIR_ROOT, eval_root=EVAL_ROOT,
                 plot_root=PLOT_ROOT):
        self.cache_dir = cache_dir
        self.checkpoint_dir_root = checkpoint_dir_root
        self.eval_root = eval_root
        self.plot_root = plot_root

    def __call__(self, run_dir, restart):
        checkpoint_dir = os.path.normpath(os.path.join(self.checkpoint_dir_root, run_dir))
        if restart is False:
            pass
        elif restart is True:
            clear_checkpoints(checkpoint_dir)  # Start from scratch (delete checkpoints)
        else:  # some string whose cp folder is copied
            clear_checkpoints(checkpoint_dir)  # copytree() requires dest not to exist
            source = "%s%s/" % (self.checkpoint_dir_root, restart)
            shutil.copytree(source, checkpoint_dir)
        plot_dir = os.path.normpath(os.path.join(self.plot_root, run_dir))
        eval_dir = os.path.normpath(os.path.join(self.eval_root, run_dir))
        return ExperimentDir(run_dir=run_dir, checkpoint_dir=checkpoint_dir, eval_dir=eval_dir, plot_dir=plot_dir)


class Struct:
    pass


def initialize(run_name, restart=False):
    """ Initialize checkpoint folder.
    - restart=False: start from existing checkpoint
    - restart=True: clear checkpoint
    - restart="string": start from another existing checkpoint
    """
    flags = Struct()
    flags.run_name = run_name
    flags.cache_dir = '/tmp/tensorflow_cache/'
    flags.checkpoint_dir_root = '/tmp/tensorflow/'
    flags.checkpoint_dir = "%s%s/" % (flags.checkpoint_dir_root, run_name)
    flags.plots_dir = '/tmp/plots/%s/' % (run_name,)
    flags.restart = restart
    if restart is False:
        pass
    elif restart is True:
        clear_checkpoints(flags)  # Start from scratch (delete checkpoints)
    else:  # some string whose cp folder is copied
        clear_checkpoints(flags)  # copytree() requires dest not to exist
        source = "%s%s/" % (flags.checkpoint_dir_root, restart)
        shutil.copytree(source, flags.checkpoint_dir)
    return flags


def cp_folder(bucket_name, src, target, erase=True, accept=all, display=True):
    s3_client = boto3.client('s3')
    s3 = boto3.resource("s3")
    if erase:
        bucket = s3.Bucket(bucket_name)
        keys = bucket.objects.filter(Prefix=target)
        for key in keys:
            if display:
                print("cp_folder deleting", key)
            key.delete()

    for root, _, content in os.walk(src):
        for c in content:  # list of files, without subfolders
            if accept is not None and not accept(c):
                continue
            # Upload the file to S3
            dest = os.path.normpath(os.path.join(root.replace(src, target, 1), c))
            src_path = os.path.normpath(os.path.join(root, c))
            if display:
                print("cp_folder", src_path, "->", bucket, dest)
            s3_client.upload_file(src_path, bucket_name, dest, ExtraArgs={"ServerSideEncryption": "AES256"})


def log_to_file(filename='tensorflow.log', level=logging.DEBUG, impl=1, remove_stdout=True):
    # https://stackoverflow.com/questions/40559667/how-to-redirect-tensorflow-logging-to-a-file
    # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/SO_JRts-VIs

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    open(filename, 'w').close()  # 'w'  erases content if it exists

    if impl == 1:
        log = logging.getLogger('tensorflow')  # get TF logger
        if remove_stdout:
            lh_stdout = log.handlers[0]
            # assert lhStdout.stream.name == '<stderr>', str(lhStdout.stream.name)
            # '<stdout>' on venom !
            log.removeHandler(lh_stdout)
        log.setLevel(level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create file handler
        fh = logging.FileHandler(filename)
        fh.setLevel(level), fh.setFormatter(formatter)
        log.addHandler(fh)
        return log
    elif impl == 2:
        tf.logging._logger.basicConfig(filename=filename, level=level)
        return tf.logging._logger


def cp_plots(name, venom_bucket_name):
    cp_folder(venom_bucket_name,
              os.path.normpath(os.path.join("/tmp/plots", name)),
              os.path.normpath(os.path.join("plots", name)),
              display=False)


def cp_checkpoints(name, venom_bucket_name):
    cp_folder(venom_bucket_name,
              os.path.normpath(os.path.join("/tmp/tensorflow", name)),
              os.path.normpath(os.path.join("checkpoints", name)),
              display=False)


def cp_evaluations(name, venom_bucket_name):
    cp_folder(venom_bucket_name,
              os.path.normpath(os.path.join("/tmp/tensorflow_eval", name)),
              os.path.normpath(os.path.join("evaluations", name)),
              display=False)


def cp_checkpoints_and_logs(name, venom_bucket_name):
    cp_checkpoints(name, venom_bucket_name)
    cp_folder(venom_bucket_name, "./log_dir/", "tf_logs/%s/" % (name,))


class S3SavingListener(tf.train.CheckpointSaverListener):
    def __init__(self, callback):
        self._callback = callback

    def before_save(self, session, global_step_value):
        print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
        self._callback()
        print('Done writing checkpoint.')

    def end(self, session, global_step_value):
        self._callback()
        print('Done with the session.')


class MatplotlibSummaryOpFactory:
    '''
    Code for generating a tensorflow image summary of a custom matplotlib plot.
    Usage: matplotlib_summary(plotting_function, argument1, argument2, ..., name="summary name")
    plotting_function is a function which take the matplotlib figure as the first argument and numpy
      versions of argument1, ..., argumentn as the additional arguments and draws the matplotlib plot on the figure
    matplotlib_summary creates and returns a tensorflow image summary
    https://gist.github.com/kkleidal/c88e033193edf92d4027943e49b27d96

    matplotlib_summary = MatplotlibSummaryOpFactory()
    def plt_mnist(f, digit):
        # f is the matplotlib figure
        # digit is a numpy version of the argument passed to matplotlib_summary
        f.gca().imshow(np.squeeze(digit, -1))
        f.gca().set_title("A random MNIST digit")

    digit = tf.random_normal([28, 28, 1])
    summary = matplotlib_summary(plt_mnist, digit, name="mnist-summary")
    all_summaries = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(".")
    with tf.Session() as sess:
        summ = sess.run(all_summaries)
        summary_writer.add_summary(summ, global_step=0)
    '''
    def __init__(self):
        self.counter = 0

    def _wrap_pltfn(self, plt_fn, max_outputs=1):
        def plot(*args):
            fig = Figure()
            canvas = FigureCanvas(fig)
            args = [fig] + list(args)
            plt_fn(*args)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            fig.clear()
            buf.seek(0)
            im = imageio.imread(buf)
            buf.close()
            return im

        def multi_plot(*args):
            images = []
            figs = []
            for _ in range(max_outputs):
                fig = Figure()
                canvas = FigureCanvas(fig)
                figs.append(fig)
            args = [figs] + list(args)
            plt_fn(*args)
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                fig.clear()
                buf.seek(0)
                images.append(imageio.imread(buf))
                buf.close()
            return images

        return plot if max_outputs == 1 else multi_plot

    def __call__(self, plt_fn, *args, name=None, max_outputs=1, family='image'):
        if name is None:
            self.counter += 1
            name = "matplotlib-summary_%d" % self.counter
        image_tensors = tf.py_func(self._wrap_pltfn(plt_fn, max_outputs=max_outputs), args,
                                   tf.uint8 if max_outputs == 1 else [tf.uint8]*max_outputs)
        if max_outputs == 1:
            image_tensors.set_shape([None, None, 4])
            image_tensors = tf.expand_dims(image_tensors, 0)
        else:
            image_tensors = tf.stack(image_tensors, axis=0)
            image_tensors.set_shape([None, None, None, 4])
        return tf.summary.image(name, image_tensors, max_outputs=max_outputs, family=family,
                                collections=[GraphKeys.SUMMARIES, IMAGE_SUMMARIES_KEY])


matplotlib_summary = MatplotlibSummaryOpFactory()

if __name__ == '__main__':
    print(1+1)
