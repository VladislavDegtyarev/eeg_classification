import collections
from typing import Any

import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
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
        ListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        Any,
        dict,
        list,
        int,
        float,
        str,
        bool,
        tuple,
        set,
        frozenset,
        collections.defaultdict,
        collections.OrderedDict,
    ])


def patch_torch_load_for_checkpoints() -> None:
    """Patch torch.load and lightning_fabric to use weights_only=False for checkpoint loading.
    
    This is needed for PyTorch 2.6+ where weights_only defaults to True.
    Patches are applied globally to allow checkpoint loading without errors.
    """
    # Patch torch.load
    _original_torch_load = torch.load
    
    def _patched_torch_load(*args, **kwargs):
        """Patched torch.load that uses weights_only=False for checkpoint loading."""
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    torch.load = _patched_torch_load
    
    # Also patch lightning_fabric if available
    try:
        from lightning_fabric.utilities import cloud_io
        _original_pl_load = cloud_io._load
        
        def _patched_pl_load(*args, **kwargs):
            """Patched lightning_fabric._load that uses weights_only=False."""
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_pl_load(*args, **kwargs)
        
        cloud_io._load = _patched_pl_load
    except ImportError:
        pass


# Register on module import
register_torch_safe_globals()
patch_torch_load_for_checkpoints()
