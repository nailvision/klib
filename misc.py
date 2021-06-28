from typing import Any, Hashable
import os
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
import wandb


class CustomWandbLogger(WandbLogger):
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Overwrite to enable saving without directory structure"""
        # upload all checkpoints from saving dir
        if self._log_model:
            save_glob = os.path.join(self.save_dir, "*.ckpt")
            wandb.save(save_glob, os.path.dirname(save_glob))


class kdict(dict):
    """Wrapper around the native dict class that allows access via dot syntax and JS-like behavior for KeyErrors."""

    def __getattr__(self, key: Hashable) -> Any:
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: Hashable, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Hashable) -> None:
        del self[key]

    def __dir__(self):
        return self.keys()

