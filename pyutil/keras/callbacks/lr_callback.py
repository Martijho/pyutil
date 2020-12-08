try:
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras import backend as K
except Exception:
    from keras.callbacks import TensorBoard
    from keras import backend as K


class LRTensorBoard(TensorBoard):
    """
    An extension to the TensorBoard callback where logs is updated with the optimizers learningrate
    before the native on_epoch_end() method is called.

    This logs learning rate to TensorBoard alongside the other metrics

    """
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
