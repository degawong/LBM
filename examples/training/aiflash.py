import datetime
import logging
import os
import random
import re
import shutil
from typing import List, Optional

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["HF_DEBUG"] = "1"

os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
# os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
# os.environ["HF_HOME"] = "/media/83nvme/wgq2719/dega/huggingface/"

# unet_backbone_signature = "runwayml/stable-diffusion-v1-5"
# vae_backbone_signature = "runwayml/stable-diffusion-v1-5"
# scheduler_backbone_signature = "stabilityai/stable-diffusion-xl-base-1.0"

import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger
from lbm.trainer.utils import StateDictAdapter

from diffusers.models import UNet2DConditionModel

def get_model(
    unet_backbone_signature: str = "runwayml/stable-diffusion-v1-5",
    vae_backbone_signature: str = "",
    scheduler_backbone_signature: str = "",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_interval",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    mask_key: str = "mask",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
):

    conditioners = []

    # Load pretrained model as base
    pipe = StableDiffusionPipeline.from_pretrained(
        unet_backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    # denoiser = DiffusersUNet2DCondWrapper.from_config(
    #     pipe.unet.config
    # ).to(torch.bfloat16)

    denoiser = UNet2DConditionModel.from_config(
        unet_backbone_signature,  
        subfolder= "unet"
    ).to(torch.bfloat16)

    state_dict = pipe.unet.state_dict()
    denoiser.load_state_dict(
        state_dict,
        strict=True,
    )

    del pipe

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=vae_backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        scheduler_backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        scheduler_backbone_signature,
        subfolder="scheduler",
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model

def get_filter_mappers():
    filters_mappers = [
        KeyFilter(
            KeyFilterConfig(
                keys=["jpg", "normal_aligned.png", "mask.png"]
            )
        ),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "jpg": "image",
                            "normal_aligned.png": "normal",
                            "mask.png": "mask",
                        }
                    )
                ),

                # image
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["ToTensor"],
                        transforms_kwargs=[
                            {},
                        ],
                    )
                ),

                # normal
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="normal",
                        transforms=["ToTensor"],
                        transforms_kwargs=[
                            {},
                        ],
                    )
                ),

                # mask
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="mask",
                        transforms=["ToTensor", "Normalize"],
                        transforms_kwargs=[
                            {},
                            {
                                "mean": 0.0,
                                "std": 1.0,
                            },
                        ],
                    )
                ),

                RescaleMapper(
                    RescaleMapperConfig(key="image")
                ),
                RescaleMapper(
                    RescaleMapperConfig(key="normal")
                ),
            ],
        ),
    ]

    return filters_mappers

def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
):

    # TRAIN
    train_filters_mappers = get_filter_mappers()

    # unbrace urls
    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(train_shards_path_or_url)
        )

    # shuffle shards
    random.shuffle(train_shards_path_or_urls_unbraced)

    # data config
    data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=20,
        shuffle_before_split_by_workers_buffer_size=20,
        shuffle_before_filter_mappers_buffer_size=2000,
        shuffle_after_filter_mappers_buffer_size=20,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    train_data_config = data_config

    # VALIDATION
    validation_filters_mappers = get_filter_mappers()

    # unbrace urls
    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(validation_shards_path_or_url)
        )

    data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    validation_data_config = data_config

    # data module
    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )

    return data_module

def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    vae_backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    unet_backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    scheduler_backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "image",
    target_key: str = "normal",
    mask_key: str = "mask",
    wandb_project: str = "lbm-relight",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 5e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    config_yaml: dict = None,
    save_ckpt_path: str = "./checkpoints",
    log_interval: int = 100,
    resume_from_checkpoint: bool = True,
    max_epochs: int = 100,
    bridge_noise_sigma: float = 0.005,
    save_interval: int = 1000,
    # NOTE: validation only fires when (batch_idx + 1) % val_check_interval == 0
    # (pytorch_lightning/loops/training_epoch_loop.py:_should_check_val_fx), so with the
    # original hardcoded 1000 an epoch shorter than 1000 steps never validates at all.
    # Exposed as a config knob; default 1000 keeps the previous behaviour byte-for-byte.
    val_check_interval: int = 1000,
    path_config: str = None,
):

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Surface"
        + f"{os.environ.get('SLURM_JOB_ID', 0)}"
        + f"_{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}"
    )
    dir_path = f"{save_ckpt_path}/logs/{training_signature}"
    # NOTE: the default MUST be the string "0". os.environ.get(..., 0) returns int 0 and
    #  is False, so outside SLURM this whole block was silently skipped:
    # checkpoints/<run>/logs/ was never created (wandb then reported its dir as "not writable"
    # and fell back to the system temp dir), and config.yaml was never copied.
    # The LOCAL_RANK guard keeps it to one process when Lightning spawns the DDP ranks.
    if os.environ.get("SLURM_PROCID", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0":
        os.makedirs(dir_path, exist_ok=True)
        if path_config is not None:
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature
        os.makedirs(dir_path, exist_ok=True)
        if path_config is not None:
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature

    model = get_model(
        vae_backbone_signature=vae_backbone_signature,
        unet_backbone_signature=unet_backbone_signature,
        scheduler_backbone_signature=scheduler_backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
    )

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["image", "normal", "mask"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
        os.path.exists(save_ckpt_path)
        and resume_from_checkpoint
        and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")

    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    # Ignore parameters unused during training
    ignore_states = []
    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in ["denoiser."]:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
        if ignore:
            ignore_states.append(param)

    # FSDP Strategy
    strategy = FSDPStrategy(
        auto_wrap_policy=ModuleWrapPolicy(
            [
                UNet2DConditionModel,
                BasicTransformerBlock,
                ResnetBlock2D,
                # torch.nn.Conv2d,
            ]
        ),
        activation_checkpointing_policy=ModuleWrapPolicy(
            [
                BasicTransformerBlock,
                ResnetBlock2D,
            ]
        ),
        sharding_strategy="SHARD_GRAD_OP",
        # sharding_strategy="SHARD_GRAD_OP",
        ignored_states=ignore_states,
    )

    devices_count = len(os.environ.get("CUDA_VISIBLE_DEVICES","").strip().split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES","").strip() else 0

    trainer = Trainer(
        accelerator="gpu",
        # mark_to_do
        devices=devices_count,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, broadcast_buffers=False) if devices_count > 1 else "auto",
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=False, name=run_name, save_dir=save_ckpt_path
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_train_steps=save_interval,
                save_last=True,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        val_check_interval=val_check_interval,
        max_epochs=max_epochs,
    )
    for n, p in pipeline.model.named_parameters():
        p.requires_grad = n.startswith("denoiser.")

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)

def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)

if __name__ == "__main__":
    # test()
    # test_denoiser()
    import sys
    DEFAULT_CFG = "/media/83nvme/wgq2719/MyCode/LBM/examples/training/config/aiflash.yaml"
    if len(sys.argv) > 1:
        # 命令行传参时使用传入路径
        fire.Fire(main_from_config)
    else:
        # 无命令行参数，使用内置默认路径
        main_from_config(DEFAULT_CFG)
