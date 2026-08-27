import logging
from typing import Callable, List, Union

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import webdataset as wds
from webdataset import DataPipeline

from ..filters import BaseFilter, FilterWrapper
from ..mappers import BaseMapper, MapperWrapper
from .collation_fn import custom_collation_fn
from .datasets_config import DataModuleConfig

logger = logging.getLogger(__name__)


def get_dist_info():
    """
    获取当前进程的分布式信息 (rank, world_size)。

    未初始化进程组时返回 (0, 1)，因此单卡/非分布式场景行为不变。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def get_worker_info():
    """
    获取当前 DataLoader worker 信息 (worker_id, num_workers)。

    必须在迭代期间调用（worker 是 DataLoader 迭代时才 fork 的），
    因此本函数只能在 pipeline stage 的生成器体内使用，不能在 setup() 阶段调用。
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return 0, 1
    return info.id, info.num_workers


def split_by_sample(src):
    """
    样本级切分（webdataset pipeline stage）——在单个 shard 内部实现真正的数据并行。

    背景
    ----
    webdataset 原生的 ``split_by_node`` 是 **shard 粒度** 的切分（``urls[rank::world_size]``）。
    当 shard 数少于 GPU 数时，靠后的 rank 会拿到空的 shard 列表，进而：
      rank0 正常训练做梯度 ALLREDUCE，其余 rank 因数据为空直接进入 on_train_end 的 barrier
      → 同一 SequenceNumber 上集合通信类型不一致 → DDP 永久互等（多卡死锁）。
    在 GPU 分发平台上可用卡数随时变化，离线重新切分 tar 并不灵活，因此改为在数据模块内在线切分。

    做法
    ----
    所有 (rank, worker) 共读**相同的完整 shard 流**，但各自只保留属于自己的样本::

        split_id     = rank * num_workers + worker_id
        total_splits = world_size * num_workers
        保留条件      : global_sample_index % total_splits == split_id

    因此单个 tar 内的大量样本会被均匀分派到各卡，**样本互不重复**，真正获得数据并行收益。

    等量保证（关键，用于杜绝死锁）
    ------------------------------
    只有当一组 ``total_splits`` 个样本**全部到齐**时，才产出本 split 暂存的那一个样本；
    流末尾不完整的一组整体丢弃。于是每个 split 严格得到 ``floor(S / total_splits)`` 个样本
    （S 为总样本数），各 rank 样本数完全相等 → batch 数完全相等 → 不会再出现
    某个 rank 提前结束 epoch 而卡在 barrier 的情况。额外内存开销仅为 1 个未解码样本。

    放置位置
    --------
    必须插在 ``tarfile_to_samples`` 之后、``decode``/mappers 之前：未被选中的样本只经过
    tar 解析（读字节），不会进入图像解码与预处理，因此 CPU 开销不重复，只有磁盘读取会放大。

    一致性前提
    ----------
    各 rank 看到的 shard 序列必须完全一致，否则全局样本索引含义不同会造成样本重复/丢失。
    调用方需保证：使用确定性的 shard 顺序，且不在本 stage 之前做 shard 级 shuffle。
    """
    rank, world_size = get_dist_info()
    worker_id, num_workers = get_worker_info()

    total_splits = world_size * num_workers
    split_id = rank * num_workers + worker_id

    # 单进程单 worker：无需切分，避免不必要的尾部丢弃
    if total_splits <= 1:
        yield from src
        return

    pending = None
    for index, sample in enumerate(src):
        position = index % total_splits
        if position == split_id:
            pending = sample
        # 一组样本已完整到齐，此时产出才能保证各 split 严格等量
        if position == total_splits - 1 and pending is not None:
            yield pending
            pending = None
    # 尾部不完整的一组整体丢弃（最多损失 total_splits - 1 个样本），换取各 rank 步数严格一致


class DataPipeline:
    """
    DataPipeline class for creating a dataloader from a single configuration

    Args:

        config (DataModuleConfig):
            Configuration for the dataset

        filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the dataset. These will be sequentially applied.

        batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        config: DataModuleConfig,
        filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ],
        batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
    ):
        self.config = config
        self.shards_path_or_urls = config.shards_path_or_urls
        self.filters_mappers = filters_mappers
        self.batched_filters_mappers = batched_filters_mappers or []

        if filters_mappers is None:
            filters_mappers = []

        # set processing pipeline
        self.processing_pipeline = [wds.decode(config.decoder, handler=config.handler)]
        self.processing_pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=filters_mappers,
                handler=config.handler,
            )
        )

    def _add_filters_mappers(
        self,
        filters_mappers: List[
            Union[
                FilterWrapper,
                MapperWrapper,
            ]
        ],
        handler: Callable = wds.warn_and_continue,
    ) -> List[Union[FilterWrapper, MapperWrapper]]:
        tmp_pipeline = []
        for filter_mapper in filters_mappers:
            if isinstance(filter_mapper, FilterWrapper) or isinstance(
                filter_mapper, BaseFilter
            ):
                tmp_pipeline.append(wds.select(filter_mapper))
            elif isinstance(filter_mapper, MapperWrapper) or isinstance(
                filter_mapper, BaseMapper
            ):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            elif isinstance(filter_mapper) or isinstance(filter_mapper):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            else:
                raise ValueError("Unknown type of filter/mapper")
        return tmp_pipeline

    def setup(self):
        # ------------------------------------------------------------------
        # 在线切分策略选择：应对 GPU 分发平台上可用卡数随时变化、离线切分 tar 不灵活的场景。
        #
        # "node"   —— 原始行为，shard 粒度切分（wds.split_by_node = urls[rank::world_size]）。
        #             shard 数能被 world_size 整除时 IO 最优（每个 rank 只读自己那几个 tar）。
        # "sample" —— 样本级切分（见 split_by_sample）。shard 数不足时启用：各 rank 共读同一
        #             shard 流，但只取 index % total_splits == split_id 的样本，从而在单个 tar
        #             内部实现真正的数据并行，样本互不重复，且各 rank 严格等量、不会死锁。
        # ------------------------------------------------------------------
        _, world_size = get_dist_info()

        if isinstance(self.shards_path_or_urls, str):
            shards = [self.shards_path_or_urls]
        else:
            shards = list(self.shards_path_or_urls)
        num_shards = len(shards)
        if num_shards == 0:
            raise ValueError("shards_path_or_urls is empty, cannot build data pipeline")

        split_mode = self.config.shard_split_mode
        if split_mode == "auto":
            # shard 粒度切分要求每个 rank 都能分到等量且非空的 shard
            shard_level_ok = world_size == 1 or (
                num_shards >= world_size and num_shards % world_size == 0
            )
            split_mode = "node" if shard_level_ok else "sample"

        if split_mode == "sample":
            # 全局样本索引要求各 rank 看到完全一致的 shard 序列，否则会样本重复/丢失。
            # 训练脚本可能对 shard 列表做过 random.shuffle，这里统一排序以强制确定性。
            shard_list = sorted(shards)
        else:
            shard_list = shards

        pipeline = [wds.SimpleShardList(shard_list)]

        if split_mode == "sample":
            if world_size > 1:
                logger.info(
                    f"[DataPipeline] shard 数({num_shards}) 无法在 world_size({world_size}) 上等分，"
                    f"启用样本级在线切分：各 rank 共读同一 shard 流，按 "
                    f"index % (world_size * num_workers) 取属于自己的样本，"
                    f"样本互不重复且各 rank 严格等量。"
                )
            # 样本级切分下刻意不做 shard 级 shuffle、也不用 split_by_node / split_by_worker：
            #   - shard 级 shuffle 会让各 rank 的 shard 顺序不同，破坏全局索引一致性
            #   - split_by_node/worker 在 shard 数不足时会让部分 rank/worker 空载（死锁根因）
            # 切分器紧跟 tarfile_to_samples，位于 decode/mappers 之前，
            # 因此未被选中的样本不会进入图像解码与预处理，CPU 开销不重复。
            pipeline.append(
                wds.tarfile_to_samples(
                    handler=self.config.handler,
                    rename_files=self.config.rename_files_fn,
                )
            )
            pipeline.append(split_by_sample)
        else:
            # shuffle before split by node
            if self.config.shuffle_before_split_by_node_buffer_size is not None:
                pipeline.append(
                    wds.shuffle(
                        self.config.shuffle_before_split_by_node_buffer_size,
                        handler=self.config.handler,
                    )
                )
            # split by node
            pipeline.append(wds.split_by_node)

            # shuffle before split by workers
            if self.config.shuffle_before_split_by_workers_buffer_size is not None:
                pipeline.append(
                    wds.shuffle(
                        self.config.shuffle_before_split_by_workers_buffer_size,
                        handler=self.config.handler,
                    )
                )
            # split by worker
            pipeline.extend(
                [
                    wds.split_by_worker,
                    wds.tarfile_to_samples(
                        handler=self.config.handler,
                        rename_files=self.config.rename_files_fn,
                    ),
                ]
            )

        # shuffle before filter mappers
        if self.config.shuffle_before_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_filter_mappers_buffer_size,
                    handler=self.config.handler,
                )
            )

        # apply filters and mappers
        pipeline.extend(self.processing_pipeline)

        # shuffle after filter mappers
        if self.config.shuffle_after_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_after_filter_mappers_buffer_size,
                    handler=self.config.handler,
                ),
            )

        # 可选：固定每 epoch 样本数，作为多卡步数一致的二次保险
        if self.config.per_epoch_samples is not None:
            pipeline.append(wds.with_epoch(self.config.per_epoch_samples))

        # batching
        pipeline.append(
            wds.batched(
                self.config.per_worker_batch_size,
                collation_fn=custom_collation_fn,
            )
        )

        # apply batched transforms
        pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=self.batched_filters_mappers,
                handler=self.config.handler,
            )
        )

        # create the data pipeline
        pipeline = wds.DataPipeline(*pipeline, handler=self.config.handler)

        # set the pipeline
        self.pipeline = pipeline

    def dataloader(self):
        # return the loader

        # dl = wds.WebLoader(
        #     self.pipeline,
        #     batch_size=None,
        #     num_workers=self.config.num_workers,
        # )

        return wds.WebLoader(
            self.pipeline,
            batch_size=None,
            num_workers=self.config.num_workers,
        )


class DataModule(pl.LightningDataModule):
    """
    Main DataModule class for creating data loaders and training/evaluating models

    Args:

        train_config (DataModuleConfig):
            Configuration for the training dataset

        train_filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the training dataset. These will be sequentially applied.

        train_batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the training dataset. These will be sequentially applied.

        eval_config (DataModuleConfig):
            Configuration for the evaluation dataset

        eval_filters_mappers (List[Union[FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the evaluation dataset.These will be sequentially applied.

        eval_batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the evaluation dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        train_config: DataModuleConfig,
        train_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
        train_batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
        eval_config: DataModuleConfig = None,
        eval_filters_mappers: List[Union[FilterWrapper, MapperWrapper]] = None,
        eval_batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
    ):
        super().__init__()

        self.train_config = train_config
        self.train_filters_mappers = train_filters_mappers
        self.train_batched_filters_mappers = train_batched_filters_mappers

        self.eval_config = eval_config
        self.eval_filters_mappers = eval_filters_mappers
        self.eval_batched_filters_mappers = eval_batched_filters_mappers

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """

        # train pipeline
        self.train_pipeline = DataPipeline(
            config=self.train_config,
            filters_mappers=self.train_filters_mappers,
            batched_filters_mappers=self.train_batched_filters_mappers,
        )
        self.train_pipeline.setup()

        # eval pipeline
        if self.eval_config is not None:
            self.eval_pipeline = DataPipeline(
                config=self.eval_config,
                filters_mappers=self.eval_filters_mappers,
                batched_filters_mappers=self.eval_batched_filters_mappers,
            )
            self.eval_pipeline.setup()

    def train_dataloader(self):

        # data = self.train_pipeline.dataloader()

        return self.train_pipeline.dataloader()

    def val_dataloader(self):
        
        return self.eval_pipeline.dataloader()
