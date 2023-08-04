class EarlyStopper:
    """
    Implements early stopping.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        :param patience: int, default=1
            Number of epochs with no improvement needed to early stop.
        :param min_delta: float, default=0
            Minimum val loss reduction needed to count as improvement.
        """
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float("inf")


    def early_stop(self, val_loss):
        """
        Checks whether we should early stop after the last epoch.

        :param val_loss: float
            Last epoch val loss.

        :return: bool
            True iff we should early stop.
        """

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

    def load_state_dict(self, state_dict):
        
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.counter = state_dict["counter"]
        self.min_val_loss = state_dict["min_val_loss"]
    

    def state_dict(self):
        
        state_dict = {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "min_val_loss": self.min_val_loss
        }

        return state_dict
    


class BestTracker:
    """
    Tracks best validation loss.
    """

    def __init__(self, min_delta=0):
        """
        :param min_delta: float, default=0
            Minimum val loss reduction needed to count as improvement.
        """
        
        self.min_delta = min_delta
        self.min_val_loss = float("inf")


    def is_best(self, val_loss):
        """
        Checks whether the last epoch had the best model.

        :param val_loss: float
            Last epoch val loss.

        :return: bool
            True iff the last epoch had the best model.
        """

        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            return True
        return False
    

    def load_state_dict(self, state_dict):
        
        self.min_delta = state_dict["min_delta"]
        self.min_val_loss = state_dict["min_val_loss"]
    

    def state_dict(self):
        
        state_dict = {
            "min_delta": self.min_delta,
            "min_val_loss": self.min_val_loss
        }

        return state_dict
