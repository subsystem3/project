import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class Metrics:
    """
    A class to hold metrics for monitoring performance.

    This class is a singleton, meaning that there is only one instance of the class throughout the entire program.
    All functions decorated with @measure_time share the same Metrics object.

    Attributes:
        metrics (dict): A dictionary to store the metrics.
    """

    _instance = None

    def __new__(cls):
        """
        Overrides the default object creation method to implement the singleton pattern.

        Returns:
            Metrics: The singleton instance of the Metrics class.
        """
        if cls._instance is None:
            cls._instance = super(Metrics, cls).__new__(cls)
            cls._instance.metrics = {}
        return cls._instance

    def __init__(self):
        """Initializes the Metrics class with an empty dictionary."""
        self.metrics: dict[str, float] = {}

    def add_metric(self, key: str, value: float) -> None:
        """
        Adds a new metric to the dictionary or appends to an existing one.

        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.
        """
        if key in self.metrics:
            self.metrics[key] += value
        else:
            self.metrics[key] = value

    def get_metrics(self) -> dict[str, float]:
        """
        Returns the dictionary of metrics.

        Returns:
            dict: The dictionary of metrics.
        """
        return self.metrics

    def __str__(self) -> str:
        """Returns a string representation of the metrics."""
        return "\n".join(f"{k}: {v}" for k, v in self.metrics.items())

    def __getitem__(self, key: str) -> float:
        """
        Returns the value of the metric with the given key.

        Args:
            key (str): The key of the metric.

        Returns:
            float: The value of the metric.
        """
        return self.metrics.get(key)

    def __setitem__(self, key: str, value: float) -> None:
        """
        Sets the value of the metric with the given key.

        Args:
            key (str): The key of the metric.
            value (float): The value of the metric.
        """
        self.metrics[key] = value

    def __len__(self) -> int:
        """
        Returns the number of metrics.

        Returns:
            int: The number of metrics.
        """
        return len(self.metrics)

    def __contains__(self, key: str) -> bool:
        """
        Checks if the metric with the given key exists.

        Args:
            key (str): The key of the metric.

        Returns:
            bool: True if the metric exists, False otherwise.
        """
        return key in self.metrics

    def clear_metrics(self) -> None:
        """Clears all the metrics."""
        self.metrics.clear()

    def remove_metric(self, key: str) -> None:
        """
        Removes the metric with the given key.

        Args:
            key (str): The key of the metric.
        """
        if key in self.metrics:
            del self.metrics[key]

    def update_metric(self, key: str, value: float) -> None:
        """
        Updates the value of the metric with the given key.

        Args:
            key (str): The key of the metric.
            value (float): The new value of the metric.
        """
        if key in self.metrics:
            self.metrics[key] = value

    def print_metrics(self) -> None:
        """Prints all the metrics in a nice format."""
        print("Metrics:")
        sorted_metrics = sorted(
            self.metrics.items(), key=lambda item: item[1], reverse=True
        )
        for key, value in sorted_metrics:
            print(f"  {key}: {value:.2f} seconds")
        print(f"Total time: {sum(self.metrics.values()):.2f} seconds")


def measure_time(run_name: str):
    """
    A decorator factory that measures the time taken by a function and logs it to the Metrics singleton.

    Args:
        run_name (str): The name of the run. This is used to differentiate metrics among different runs.

    Returns:
        Callable: The decorator that wraps the function with timing and logging.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        The actual decorator that wraps the function with timing and logging.

        Args:
            func (Callable): The function to be timed and logged.

        Returns:
            Callable: The wrapped function.
        """

        def wrapper(*args, **kwargs):
            """
            The wrapper function that adds timing and logging to the original function.

            Args:
                *args: The positional arguments for the original function.
                **kwargs: The keyword arguments for the original function.

            Returns:
                The result of the original function.
            """
            # START TIMER
            start_time = time.time()
            # CALL FUNCTION
            result = func(*args, **kwargs)
            # CALCULATE ELAPSED TIME
            total_time = time.time() - start_time
            # LOG RESULT
            metrics.add_metric(f"{run_name}", total_time)
            print(f"{run_name}...{total_time:.2f} seconds")
            return result

        return wrapper

    return decorator
