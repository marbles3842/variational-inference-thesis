import csv
from typing import Optional, List, Dict


class MetricsLogger:
    """
    A simple CSV logger for tracking metrics across training epochs.

    Args:
        filepath (str): Path to the CSV file for logging.

    Usage:
        with MetricsLogger("metrics.csv") as logger:
            logger.update("train", "loss", 0.5)
            logger.update("val", "accuracy", 0.8)
            logger.end_epoch()

    Methods:
        update(split, metric, value): Store a metric value for the current epoch.
        end_epoch(): Finalize and write current metrics to CSV.
        latest(split, metric): Get the latest stored value for a given metric.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.fieldnames: Optional[List[str]] = None
        self._file = None
        self._writer = None
        self._current_row: Dict[str, float] = {}
        self._epoch = 0

    def __enter__(self):
        self._file = open(self.filepath, "w", newline="")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def update(self, split: str, metric: str, value: float):
        key = f"{split}_{metric}"
        self._current_row[key] = value

    def end_epoch(self):
        self._epoch += 1
        row = {"epoch": self._epoch, **self._current_row}
        if self.fieldnames is None:
            self.fieldnames = ["epoch"] + sorted(self._current_row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        self._current_row = {}

    def latest(self, split: str, metric: str):
        key = f"{split}_{metric}"
        return self._current_row.get(key, None)
