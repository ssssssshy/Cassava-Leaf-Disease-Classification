class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        # assert заменён на raise ValueError — assert удаляется при python -O,
        # что приводит к silent-багу (невалидный mode молча ломает логику сравнения)
        if mode not in ("max", "min"):
            raise ValueError(
                f"mode должен быть 'max' или 'min', получено: {mode}"
            )

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float):
        if self.best_score is None:
            self.best_score = current_score
            return

        if self._is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_better(self, score: float, best: float) -> bool:
        if self.mode == "max":
            return score > best + self.min_delta
        return score < best - self.min_delta
