from collections import defaultdict

try:
    from tensorflow.keras.callbacks import Callback
except Exception:
    from keras.callbacks import Callback


class CallbackWrapper(Callback):
    """
    Wrapper object for calls that are to be made as part of a keras training pipeline.

    To use this functionality, provide this callback-wrapper with lists of objects or callable functions.
    The possible named parameters to use is named according to keras callback-convention, and all callables provided
    in one or more lists will be called in the cooresponding method.

    Input-arguments to callable function must match those provided to the matching keras-callback function
    I.E:

    def wrapped_on_epoch_end_function(epoch, logs=None):
        print('Printed during callback in on epoch end')

    model.fit(
        ...
        callbacks=[
            CallbackWrapper(on_epoch_end=wrapped_on_epoch_end_function)
        ]
    )
    """
    def __init__(
            self,
            **kwargs
    ):
        self.callables = defaultdict(list)
        for kw, funcs in kwargs.items():
            if funcs is None:
                continue
            if hasattr(self, kw) and hasattr(getattr(self, kw), '__call__'):
                if type(funcs) != list:
                    funcs = [funcs]

                for f in funcs:
                    if hasattr(f, '__call__'):
                        self.callables[kw].append(f)

        def get_wrapper_function(funcs):
            def wrapper(*args, **kwargs):
                for c in funcs:
                    c(*args, **kwargs)
            return wrapper

        for name, funcs in self.callables.items():
            setattr(self, name, get_wrapper_function(funcs))

        super().__init__()

