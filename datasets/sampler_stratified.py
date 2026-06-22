"""分层随机身份采样器 - 解决 CombinedReID 数据集不平衡问题。"""
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class StratifiedRandomIdentitySampler(Sampler):
    """分层随机身份采样器，支持三种采样策略解决数据集不平衡问题。"""

    def __init__(
        self,
        data_source,
        batch_size,
        num_instances,
        strategy='proportional',
        pid_to_dataset_id=None,
        num_datasets=9,
        custom_weights=None,
        min_pids_per_dataset=1,
        oversample_small=True,
        small_dataset_threshold=1000,
        small_dataset_factor=2.0,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.strategy = strategy.lower()
        self.pid_to_dataset_id = pid_to_dataset_id
        self.num_datasets = num_datasets
        self.min_pids_per_dataset = min_pids_per_dataset
        self.oversample_small = oversample_small
        self.small_dataset_threshold = small_dataset_threshold
        self.small_dataset_factor = small_dataset_factor

        if self.strategy not in ['uniform', 'proportional', 'custom']:
            raise ValueError(f"Unknown strategy: {strategy}")

        if self.strategy == 'custom':
            if custom_weights is None:
                raise ValueError("custom_weights must be provided for 'custom' strategy")
            self.custom_weights = np.array(custom_weights) / np.sum(custom_weights)
        else:
            self.custom_weights = None

        self.index_dic = defaultdict(list)
        self.dataset_pids = defaultdict(list)
        self.dataset_img_counts = defaultdict(int)

        for index, (_, pid, _, _, dataset_id) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if pid not in self.dataset_pids[dataset_id]:
                self.dataset_pids[dataset_id].append(pid)
            self.dataset_img_counts[dataset_id] += 1

        self.pids = list(self.index_dic.keys())
        self.available_datasets = sorted(self.dataset_pids.keys())
        self._compute_dataset_weights()
        self.length = self._estimate_length()

        print(f"StratifiedSampler: strategy={strategy}, batch_size={batch_size}, num_pids_per_batch={self.num_pids_per_batch}")

    def _compute_dataset_weights(self):
        if self.strategy == 'uniform':
            self.dataset_weights = np.ones(self.num_datasets) / self.num_datasets
        elif self.strategy == 'proportional':
            weights = np.zeros(self.num_datasets)
            for d in self.available_datasets:
                count = self.dataset_img_counts[d]
                if self.oversample_small and count < self.small_dataset_threshold:
                    weights[d] = count * self.small_dataset_factor
                else:
                    weights[d] = count
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(self.num_datasets) / self.num_datasets
            self.dataset_weights = weights
        elif self.strategy == 'custom':
            self.dataset_weights = self.custom_weights

    def _estimate_length(self):
        length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            length += num - num % self.num_instances
        return length

    def _compute_pid_quota(self):
        base_quota = np.floor(self.dataset_weights * self.num_pids_per_batch).astype(int)
        remaining = self.num_pids_per_batch - base_quota.sum()
        if remaining > 0:
            frac = self.dataset_weights * self.num_pids_per_batch - base_quota
            sorted_indices = np.argsort(-frac)
            for i in range(int(remaining)):
                dataset_id = sorted_indices[i]
                if dataset_id in self.available_datasets:
                    base_quota[dataset_id] += 1

        for d in self.available_datasets:
            max_possible = len(self.dataset_pids[d])
            base_quota[d] = min(max(base_quota[d], self.min_pids_per_dataset), max_possible)

        total_quota = base_quota.sum()
        if total_quota > self.num_pids_per_batch:
            ratio = self.num_pids_per_batch / total_quota
            base_quota = np.floor(base_quota * ratio).astype(int)
            for d in self.available_datasets:
                if len(self.dataset_pids[d]) > 0 and base_quota[d] == 0:
                    base_quota[d] = 1

        return base_quota

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        dataset_batches = {}
        for d in self.available_datasets:
            pids_in_dataset = copy.deepcopy(self.dataset_pids[d])
            random.shuffle(pids_in_dataset)
            dataset_batches[d] = []
            for pid in pids_in_dataset:
                if batch_idxs_dict[pid]:
                    dataset_batches[d].append((pid, batch_idxs_dict[pid].pop(0)))
                    if not batch_idxs_dict[pid]:
                        batch_idxs_dict.pop(pid)

        final_idxs = []
        while True:
            quota = self._compute_pid_quota()
            selected_pids = []
            for d in self.available_datasets:
                num_pids = int(quota[d])
                if num_pids > 0 and dataset_batches[d]:
                    if len(dataset_batches[d]) <= num_pids:
                        selected = dataset_batches[d]
                        dataset_batches[d] = []
                    else:
                        indices = random.sample(range(len(dataset_batches[d])), num_pids)
                        selected = [dataset_batches[d][i] for i in sorted(indices, reverse=True)]
                        for i in sorted(indices, reverse=True):
                            del dataset_batches[d][i]
                    selected_pids.extend(selected)

            if not selected_pids:
                break

            random.shuffle(selected_pids)
            for pid, batch_idxs in selected_pids:
                final_idxs.extend(batch_idxs)
                if pid in batch_idxs_dict and batch_idxs_dict[pid]:
                    dataset_id = self.pid_to_dataset_id[pid]
                    dataset_batches[dataset_id].append((pid, batch_idxs_dict[pid].pop(0)))

        return iter(final_idxs)

    def __len__(self):
        return self.length