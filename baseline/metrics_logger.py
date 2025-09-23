import csv
import fcntl
import os
import time
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


class ConcurrentMetricsLogger:
    """
    A thread-safe CSV logger for tracking metrics across multiple concurrent processes.

    Args:
        filepath (str): Path to the CSV file for logging.
        max_retries (int): Maximum number of retries for file operations.
        retry_delay (float): Delay between retries in seconds.

    Usage:
        with ConcurrentMetricsLogger("shared_metrics.csv") as logger:
            logger.update("test", "loss", 0.5, seed=42, job_id=1)
            logger.update("test", "accuracy", 0.8, seed=42, job_id=1)
            logger.write_row()
    """

    def __init__(self, filepath: str, max_retries: int = 10, retry_delay: float = 0.1):
        self.filepath = filepath
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._current_row: Dict[str, float] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, split: str, metric: str, value: float, seed: int, job_id: int):
        key = f"{split}_{metric}"
        self._current_row[key] = value
        self._current_row["seed"] = seed
        self._current_row["job_id"] = job_id

    def write_row(self):
        if not self._current_row:
            return

        for attempt in range(self.max_retries):
            try:
                self._write_with_lock()
                self._current_row = {}
                return
            except (IOError, OSError) as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2**attempt))

    def _write_with_lock(self):
        file_exists = os.path.exists(self.filepath)

        with open(self.filepath, "a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            try:
                needs_header = not file_exists or os.path.getsize(self.filepath) == 0

                if needs_header:
                    fieldnames = ["seed", "job_id"] + sorted(
                        [
                            k
                            for k in self._current_row.keys()
                            if k not in ["seed", "job_id"]
                        ]
                    )
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                else:
                    with open(self.filepath, "r") as read_f:
                        reader = csv.reader(read_f)
                        fieldnames = next(reader, [])

                    if fieldnames:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                    else:
                        fieldnames = ["seed", "job_id"] + sorted(
                            [
                                k
                                for k in self._current_row.keys()
                                if k not in ["seed", "job_id"]
                            ]
                        )
                        writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writerow(self._current_row)
                f.flush()
                os.fsync(f.fileno())

            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
