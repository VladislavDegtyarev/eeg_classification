import collections
from typing import Any

import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.nodes import AnyNode

from src.utils.env_utils import (
    collect_random_states,
    log_gpu_memory_metadata,
    set_max_threads,
    set_seed,
)
from src.utils.metadata_utils import log_metadata
from src.utils.pylogger import get_pylogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.saving_utils import save_predictions, save_state_dicts
from src.utils.utils import (
    close_loggers,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_plugins,
    log_hyperparameters,
    register_custom_resolvers,
    save_file,
    task_wrapper,
)


def register_torch_safe_globals() -> None:
    """Register safe globals for torch.load with weights_only=True.

    This is needed for PyTorch 2.6+ where weights_only defaults to True.
    Must be called before any checkpoint loading, including in multiprocessing spawn.
    """
    torch.serialization.add_safe_globals([
        DictConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        Any,
        dict,
        collections.defaultdict,
        collections.OrderedDict,
    ])


# Register on module import
register_torch_safe_globals()
