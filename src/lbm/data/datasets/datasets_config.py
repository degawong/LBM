from typing import Callable, List, Optional, Union

import webdataset as wds
from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class DataModuleConfig(BaseConfig):
    """
    Configuration for the DataModule

    Args:

        shards_path_or_urls (Union[str, List[str]]): The path or url to the shards. Defaults to None.
        per_worker_batch_size (int): The batch size for the dataset. Defaults to 16.
        num_workers (int): The number of workers to use. Defaults to 1.
        shuffle_before_split_by_node_buffer_size (Optional[int]): The buffer size for the shuffle before split by node. Defaults to 100.
        shuffle_before_split_by_workers_buffer_size (Optional[int]): The buffer size for the shuffle before split by workers. Defaults to 100.
        shuffle_before_filter_mappers_buffer_size (Optional[int]): The buffer size for the shuffle before filter mappers. Defaults to 1000.
        shuffle_after_filter_mappers_buffer_size (Optional[int]): The buffer size for the shuffle after filter mappers. Defaults to 1000.
        decoder (str): The decoder to use. Defaults to "pil".
        handler (Callable): A callable to handle the warnings. Defaults to wds.warn_and_continue.
        rename_files_fn (Optional[Callable[[str], str]]): A callable to rename the files. Defaults to None.
        shard_split_mode (str): 多卡数据切分粒度，用于应对 GPU 分发平台上可用卡数随时变化的场景。
            - "auto"（默认）：分片数 >= world_size 且能被 world_size 整除时走 "node"（IO 最优）；
              否则自动切换到 "sample"，避免靠后的 rank 拿到空数据集而导致 DDP 死锁。
            - "node"：原始行为，按 shard 粒度切分（wds.split_by_node，即 urls[rank::world_size]）。
              要求分片数能被 world_size 整除，否则会有 rank 空载/步数不一致的死锁风险。
            - "sample"：样本级切分。各 rank/worker 共读相同的完整 shard 流，但只保留
              global_index % total_splits == split_id 的样本，从而在**单个 shard 内部**实现真正的
              数据并行——样本互不重复，无需离线重新切分 tar。
            Defaults to "auto".
        per_epoch_samples (Optional[int]): 可选，固定每个 epoch 每个 worker 截取的样本数（wds.with_epoch），
            作为多卡步数一致的额外保险。"sample" 模式下切分器已保证各 rank 严格等量，通常无需设置。
            None=不限制。Defaults to None.
    """

    shards_path_or_urls: Union[str, List[str]] = None
    per_worker_batch_size: int = 16
    num_workers: int = 1
    shuffle_before_split_by_node_buffer_size: Optional[int] = 100
    shuffle_before_split_by_workers_buffer_size: Optional[int] = 100
    shuffle_before_filter_mappers_buffer_size: Optional[int] = 1000
    shuffle_after_filter_mappers_buffer_size: Optional[int] = 1000
    decoder: str = "pil"
    handler: Callable = wds.warn_and_continue
    rename_files_fn: Optional[Callable[[str], str]] = None
    shard_split_mode: str = "auto"
    per_epoch_samples: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.rename_files_fn is not None:
            assert callable(self.rename_files_fn), "rename_files must be a callable"
        assert self.shard_split_mode in (
            "auto",
            "node",
            "sample",
        ), f"shard_split_mode must be one of 'auto'/'node'/'sample', got {self.shard_split_mode}"
