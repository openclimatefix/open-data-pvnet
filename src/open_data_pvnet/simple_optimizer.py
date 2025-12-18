from torch import optim

class SimpleAdamW(optim.AdamW):
    """
    A wrapper class for AdamW that can be instantiated with a model object.
    
    The PVNet library expects an optimizer class that takes the model instance (self)
    as the first argument in `configure_optimizers`: `return self._optimizer(self)`.
    Standard PyTorch optimizers expect `params` (iterable of parameters).
    
    This wrapper extracts the parameters from the model and passes them to the superclass.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model.parameters(), **kwargs)
