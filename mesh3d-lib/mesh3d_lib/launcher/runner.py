import sys
import json
from typing import Callable

from loguru import logger


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def run(run_fn: Callable, config_filepath: str):
    """
    Method to run a function whose parameters are described in a json configuration file

    Parameters
    ----------
    run_fn: Callable
        The run function.
    config_filepath: str
        input configuration filepath
    """

    # Setup configuration
    with open(config_filepath, "r") as f:
        config = json.load(f)

    if "seed" in config:
        set_seed(config["seed"])
    else:
        set_seed(256942)

    # Launch function
    with logger.catch(reraise=False, exclude=KeyboardInterrupt, onerror=lambda _: sys.exit(1)):
        run_fn(config)
