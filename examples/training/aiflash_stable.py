"""Stable/diagnostic entry point for the SD1.5 FFHQ experiment.

This wrapper keeps the dataset keys used by the original experiment unchanged
(image/normal/mask), while making the prompt path, loss diagnostics and WandB
sampling policy explicit. It can be launched directly from the repository root
or by an absolute path; in both cases the repository root is added to
``sys.path`` before importing the sibling training module.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# When Python executes this file directly, sys.path[0] is
# ``examples/training`` rather than the repository root. Therefore
# ``from examples.training import aiflash`` otherwise fails with
# ModuleNotFoundError: No module named 'examples'.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from examples.training import aiflash as _base

logger = logging.getLogger(__name__)


class ConfiguredLBMModel(_base.LBMModel):
    """Use the configured SD1.5 prompt embedding instead of a hard-coded file."""

    def __init__(self, *args, prompt_embedding_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        path = prompt_embedding_path or os.environ.get("LBM_PROMPT_EMBEDDING_PATH")
        if not path:
            raise ValueError(
                "prompt_embedding_path is required; pass the embedding generated "
                "for the exact SD1.5 prompt 'flash lighting'."
            )
        path = str(Path(path).expanduser())
        if not Path(path).is_file():
            raise FileNotFoundError(f"Prompt embedding does not exist: {path}")
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        if embedding.ndim != 3 or embedding.shape[1:] != (77, 768):
            raise ValueError(
                f"Expected SD1.5 prompt embedding [1, 77, 768], got {tuple(embedding.shape)}"
            )
        self.prompt_path = path
        self.prompt_embedding = embedding.contiguous()
        logger.info("Using prompt embedding %s, shape=%s", path, tuple(embedding.shape))


class StableWandbSampleLogger(_base.WandbSampleLogger):
    """Log samples by global step, avoiding batch_idx reset/repeated images."""

    def __init__(self, log_batch_freq: int = 100):
        super().__init__(log_batch_freq=log_batch_freq)
        self._last_logged_step: Dict[str, int] = {}

    def log_samples(self, trainer, pl_module, outputs, batch, batch_idx, split="train"):
        step = int(trainer.global_step)
        if step == 0 or step % self.log_batch_freq != 0:
            return
        if self._last_logged_step.get(split) == step:
            return
        self._last_logged_step[split] = step
        # The base implementation checks batch_idx, so pass zero deliberately
        # after applying the global-step gate above.
        return super().log_samples(trainer, pl_module, outputs, batch, 0, split=split)


def _training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> dict:
    output = self.model(train_batch)
    result = {
        "loss": output["loss"],
        "latent_recon_loss": output["latent_recon_loss"].detach(),
        "pixel_recon_loss": output["pixel_recon_loss"].detach(),
        "batch_idx": batch_idx,
    }
    if self.trainer.is_global_zero:
        logger.info(
            "step=%s loss=%.6f latent=%.6f pixel=%.6f",
            self.global_step,
            float(output["loss"].detach()),
            float(output["latent_recon_loss"].detach().mean()),
            float(output["pixel_recon_loss"].detach().mean()),
        )
    return result


def _validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
    output = self.model(val_batch)
    metrics = self.model.compute_metrics(val_batch)
    return {
        "loss": output["loss"],
        "latent_recon_loss": output["latent_recon_loss"].detach(),
        "pixel_recon_loss": output["pixel_recon_loss"].detach(),
        "metrics": metrics,
    }


def main_from_config(path_config: str):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)

    prompt_path = config.pop("prompt_embedding_path", None)
    if prompt_path:
        os.environ["LBM_PROMPT_EMBEDDING_PATH"] = str(Path(prompt_path).expanduser())

    # Keep all original dataset keys untouched. The stable model receives the
    # prompt path through the environment because the legacy main signature does
    # not yet expose this argument.
    _base.LBMModel = ConfiguredLBMModel
    _base.WandbSampleLogger = StableWandbSampleLogger
    _base.TrainingPipeline.training_step = _training_step
    _base.TrainingPipeline.validation_step = _validation_step

    logging.info("Running stable AIFLASH config:\n%s", yaml.dump(config, sort_keys=False))
    _base.main(
        **config,
        config_yaml={**config, "prompt_embedding_path": prompt_path},
        path_config=path_config,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main_from_config)
