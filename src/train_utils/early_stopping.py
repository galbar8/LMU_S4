class EarlyStopping:
    """Early stop on a monitored metric (e.g., val acc)."""
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience
