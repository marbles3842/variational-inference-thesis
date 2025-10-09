import os
from typing import Dict, Any
from datetime import datetime


class TestResultsLogger:
    """
    A logger for writing test results to a file.

    Args:
        log_dir (str): Directory where the log file will be saved.
        optimizer (str): Name of the optimizer being evaluated.
        model_name (str): Name of the model being evaluated.

    Usage:
        with TestResultsLogger("logs/", "sgd", "resnet") as logger:
            for seed in seeds:
                logger.start_seed(seed)
                logger.log_seed_metrics(seed, metrics_dict)

            # Compute aggregate stats in your script
            aggregate_stats = compute_aggregate_metrics(all_metrics)
            logger.write_aggregate_results(num_seeds, aggregate_stats)

    Methods:
        start_seed(seed): Mark the beginning of evaluation for a seed.
        log_seed_metrics(seed, metrics): Write metrics for a specific seed.
        write_aggregate_results(num_seeds, aggregate_stats): Write pre-computed aggregate statistics.
    """

    def __init__(self, log_dir: str, optimizer: str, model_name: str):
        self.log_dir = log_dir
        self.optimizer = optimizer
        self.model_name = model_name
        self._file = None
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.log_dir, exist_ok=True)

        filename = f"test-{self.model_name}-{self.optimizer}-{self._timestamp}.txt"
        self.filepath = os.path.join(self.log_dir, filename)

    def __enter__(self):
        self._file = open(self.filepath, "w")
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def _write_header(self):
        """Write the header information to the file."""
        header = "=" * 60 + "\n"
        header += f"TEST RESULTS\n"
        header += f"Model: {self.model_name}\n"
        header += f"Optimizer: {self.optimizer.upper()}\n"
        header += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 60 + "\n"
        self._file.write(header)
        self._file.flush()

    def start_seed(self, seed: int):
        """Mark the beginning of evaluation for a seed."""
        msg = f"\nEvaluating model with seed {seed}...\n"
        print(msg.strip())
        self._file.write(msg)
        self._file.flush()

    def log_seed_metrics(self, seed: int, metrics: Dict[str, Any]):
        """
        Write metrics for a specific seed.

        Args:
            seed (int): The seed number.
            metrics (Dict[str, Any]): Dictionary of metric names and values.
        """
        msg = f"Seed {seed} metrics: {metrics}\n"
        print(msg.strip())
        self._file.write(msg)
        self._file.flush()

    def write_aggregate_results(
        self, num_seeds: int, aggregate_stats: Dict[str, Dict[str, float]]
    ):
        """
        Write pre-computed aggregate statistics.

        Args:
            num_seeds (int): Number of seeds evaluated.
            aggregate_stats (Dict[str, Dict[str, float]]): Dictionary mapping metric names
                to their statistics (e.g., {'accuracy': {'mean': 0.95, 'std': 0.02}})
        """
        header = "\n" + "=" * 60 + "\n"
        header += (
            f"AGGREGATE RESULTS ACROSS {num_seeds} SEEDS FOR {self.optimizer.upper()}\n"
        )
        header += "=" * 60 + "\n"
        print(header.strip())
        self._file.write(header)

        for metric_name, stats in aggregate_stats.items():
            mean = stats["mean"]
            std = stats["std"]
            msg = f"{metric_name}: {mean:.4f} ± {std:.4f}\n"
            print(msg.strip())
            self._file.write(msg)

        self._file.flush()
        print(f"\nResults saved to {self.filepath}")
