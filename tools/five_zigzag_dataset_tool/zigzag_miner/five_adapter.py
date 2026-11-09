"""Bo noi tap cau truc vao bo nao nam tac nhan hien tai.

Duong moi uu tien tap chi muc + bang du lieu moi truong. Callback cu van duoc
ho tro cho tap 0.1, nhung khong duoc xem la bang chung khop giao uoc 137.
"""
from __future__ import annotations

from typing import Callable, Iterator

from .io import iter_dataset
from .project_five_adapter import ProjectFiveDataset, iter_index_samples


def iter_five_batches(
    dataset_dir: str,
    observation_builder: Callable,
    batch_size: int = 256,
) -> Iterator[list]:
    """Bo noi cu: nguoi goi tu chiu trach nhiem tao observation dung giao uoc."""
    batch = []
    for sample in iter_dataset(dataset_dir):
        obs = observation_builder(sample)
        batch.append({"observation": obs, "sample": sample})
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_project_five_episodes(
    dataset_dir: str,
    *,
    project_root: str,
    config_path: str,
) -> Iterator[dict]:
    """Tra tung tap phim dung dung ``TradingEnvironment`` va giao uoc 137.

    ``context_observations`` chi gom qua khu truoc neo. Observation tai neo duoc
    tra rieng; tu neo ve sau phai lay observation dong tu moi truong sau moi
    hanh dong, khong dung cac observation tuong lai trung tinh lam input hoc.
    """
    dataset = ProjectFiveDataset(dataset_dir, project_root, config_path)
    for sample in iter_index_samples(dataset_dir):
        env = dataset.environment_for_sample(sample)
        context, anchor = dataset.context_and_anchor(sample)
        yield {
            "environment": env,
            "context_observations": context,
            "anchor_observation": anchor,
            "anchor_local_index": int(sample["tape_anchor_index"]) - int(sample["tape_start_index"]),
            "sample": sample,
        }
