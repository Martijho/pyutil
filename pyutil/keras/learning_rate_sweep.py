from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

try:
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import Callback, LearningRateScheduler
except Exception:
    from keras import backend as K
    from keras.callbacks import Callback, LearningRateScheduler


class GatherLRDataCallback(Callback):
    def __init__(
            self,
    ):
        self.lr = []
        self.losses = defaultdict(list)
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        assert 'loss' in logs, 'No \"loss\" found in metrics'

        self.lr.append(K.eval(self.model.optimizer.lr))
        for k, v in logs.items():
            self.losses[k].append(v)

    def _plot_metric_in_subplot(self, axes, loss, name='', y_max=None):
        axes[0].plot(self.lr[:len(loss)], loss)
        # axes[0].set_xlim([self.lr[0], self.lr[-1]])
        if y_max is not None:
            axes[0].set_ylim([0, y_max])
        axes[0].set_xlabel('Learning rate')
        axes[0].set_ylabel(f'{name}')
        axes[0].grid('on')

        axes[1].plot(loss)
        axes[1].set_xlabel('Step')
        axes[1].set_xlim([0, len(self.lr)])
        axes[1].set_ylabel(f'{name}')
        if y_max is not None:
            axes[1].set_ylim([0, y_max])
        axes[1].grid('on')

        axes[2].plot(self.lr)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Learning rate')
        axes[2].grid('on')

    def plot(self, cutoff=None):
        fig, axs = plt.subplots(nrows=len(self.losses), ncols=3)
        axs[0, 0].set_title('Loss-LR plot')
        axs[0, 1].set_title('Loss-step plot')
        axs[0, 2].set_title('LR-step plot')

        for i, (k, v) in enumerate(self.losses.items()):
            v = np.array(v)
            y_max = None

            d = np.diff(v)
            if np.max(~np.isnan(d)) > 100:
                cutoff = np.argmax((d > 100)*1)
                y_max = np.max(v[:cutoff])

            self._plot_metric_in_subplot(
                axs[i, :],
                v,
                name=k,
                y_max=y_max if cutoff is None else cutoff
            )
        plt.show()


def _get_lr_sweep(
        lr_start=1e-6,
        lr_stop=1e3,
        steps=5000,
        mode='linear'
):
    assert mode in {'linear', 'exponential'}
    lr = np.zeros(steps)
    if mode == 'linear':
        lr = np.arange(lr_start, lr_stop, (lr_stop-lr_start)/steps)
    if mode == 'exponential':
        a = lr_start
        b = np.log(lr_stop/lr_start)/steps
        lr = a*np.exp(b*(np.arange(steps)))

    def sweep(epoch):
        print('epoch:', epoch, lr[epoch])
        return lr[epoch]

    return sweep


def run_lr_sweep(
        model,
        x,
        y=None,
        lr_start=1e-6,
        lr_stop=1e2,
        steps=1000,
        mode='exponential',
        cutoff=None
):
    """
    Takes a compiler keras model and runs a learning rate sweep with some data.
    Will plot Loss and Learning Rate after completed run.

    :param model: Compiled Keras model
    :param x: either data generator or training input
    :param y: None if x is a data generator, else training targets
    :param lr_start: Lower bound for learning rate
    :param lr_stop: Higher bound for learning rate
    :param steps: Number of steps to perform learning sweep over. In effect this is the number of epochs
    with steps_per_epoch=1
    :param mode: Rate of change for learning rate. Either 'linear' or 'exponential'.
    :param cutoff: When loss explodes, the plotting will try to clip that part of the plot so smaller differences in
    loss is visible without extensive zooming. Optionally, set this argument to limit the loss-axis in all plots
    """
    data = GatherLRDataCallback()
    sweep_func = _get_lr_sweep(lr_start, lr_stop, steps, mode)
    # plt.plot([sweep_func(i) for i in range(steps)])
    #
    # plt.xlabel('steps')
    # plt.ylabel('LR')
    # plt.show()

    model.fit(
        x if y is None else (x, y),
        epochs=steps,
        steps_per_epoch=1,
        shuffle=False,
        callbacks=[
            data,
            LearningRateScheduler(sweep_func)
        ]
    )
    data.plot(cutoff=cutoff)

