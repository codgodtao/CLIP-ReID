from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class BalancedDatasetSampler(Sampler):
    """
    Balanced Dataset Sampler: 控制从各子数据集采样的比例，避免数据不平衡问题。

    算法核心思想：
    1. 将所有身份(pid)按所属数据集分组
    2. 每个batch中，按照指定的权重比例从各个数据集中选取身份
    3. 对每个选中的身份，随机选取 K 个实例

    支持的平衡模式 (balance_mode):
    - 'uniform': 每个数据集被选中的概率相同（均匀分布），有效避免大样本数据集主导
    - 'proportional': 按数据集身份数量比例采样（接近原始随机采样）
    - 'square_root': 按数据集身份数量的平方根比例采样（介于均匀和比例之间）

    Args:
    - data_source (list): list of (img_path, pid, camid, trackid)
    - batch_size (int): number of examples in a batch
    - num_instances (int): number of instances per identity in a batch
    - dataset_pid_ranges (dict): {dataset_name: (pid_start, pid_end)}，每个数据集的pid范围
    - balance_mode (str): 平衡模式，'uniform' | 'proportional' | 'square_root'
    - dataset_weights (dict or None): 自定义每个数据集的权重，balance_mode='custom'时使用
    """

    def __init__(self, data_source, batch_size, num_instances,
                 dataset_pid_ranges=None, balance_mode='uniform',
                 dataset_weights=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.balance_mode = balance_mode

        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.dataset_pids = defaultdict(list)
        if dataset_pid_ranges is not None:
            for ds_name, (pid_start, pid_end) in dataset_pid_ranges.items():
                for pid in self.pids:
                    if pid_start <= pid < pid_end:
                        self.dataset_pids[ds_name].append(pid)
            self.dataset_names = list(self.dataset_pids.keys())
        else:
            self.dataset_pids['default'] = self.pids
            self.dataset_names = ['default']

        self._compute_dataset_weights(dataset_weights)

    def _compute_dataset_weights(self, dataset_weights):
        if self.balance_mode == 'custom' and dataset_weights is not None:
            self.dataset_weights = []
            for name in self.dataset_names:
                self.dataset_weights.append(dataset_weights.get(name, 0.0))
            total = sum(self.dataset_weights)
            if total > 0:
                self.dataset_weights = [w / total for w in self.dataset_weights]
            else:
                self.dataset_weights = [1.0 / len(self.dataset_names)] * len(self.dataset_names)
        elif self.balance_mode == 'proportional':
            counts = [len(self.dataset_pids[name]) for name in self.dataset_names]
            total = sum(counts)
            self.dataset_weights = [c / total for c in counts]
        elif self.balance_mode == 'square_root':
            counts = [len(self.dataset_pids[name]) for name in self.dataset_names]
            sqrt_counts = [np.sqrt(c) for c in counts]
            total = sum(sqrt_counts)
            self.dataset_weights = [c / total for c in sqrt_counts]
        else:
            self.dataset_weights = [1.0 / len(self.dataset_names)] * len(self.dataset_names)

    def _sample_pids_from_datasets(self, num_pids, available_pids_per_ds):
        selected_pids = []
        remaining = num_pids
        available_datasets = []
        available_weights = []
        for i, name in enumerate(self.dataset_names):
            if len(available_pids_per_ds[name]) > 0:
                available_datasets.append(name)
                available_weights.append(self.dataset_weights[i])
        total_w = sum(available_weights)
        if total_w == 0:
            return selected_pids
        available_weights = [w / total_w for w in available_weights]

        while remaining > 0 and len(available_datasets) > 0:
            counts_per_ds = {}
            for i, name in enumerate(available_datasets):
                expected = available_weights[i] * num_pids
                counts_per_ds[name] = max(1, int(round(expected)))

            total_expected = sum(counts_per_ds.values())
            if total_expected > num_pids:
                ratio = num_pids / total_expected
                for name in counts_per_ds:
                    counts_per_ds[name] = max(1, int(counts_per_ds[name] * ratio))

            total_expected = sum(counts_per_ds.values())
            diff = num_pids - total_expected
            if diff != 0 and len(available_datasets) > 0:
                sorted_ds = sorted(available_datasets,
                                   key=lambda n: len(available_pids_per_ds[n]),
                                   reverse=True)
                for i in range(abs(diff)):
                    ds = sorted_ds[i % len(sorted_ds)]
                    if diff > 0:
                        counts_per_ds[ds] = counts_per_ds.get(ds, 0) + 1
                    else:
                        if counts_per_ds.get(ds, 0) > 1:
                            counts_per_ds[ds] -= 1

            for name in available_datasets:
                n_select = min(counts_per_ds.get(name, 0), len(available_pids_per_ds[name]))
                if n_select > 0:
                    chosen = random.sample(available_pids_per_ds[name], n_select)
                    selected_pids.extend(chosen)
                    for pid in chosen:
                        available_pids_per_ds[name].remove(pid)

            new_available_datasets = []
            new_available_weights = []
            for i, name in enumerate(available_datasets):
                if len(available_pids_per_ds[name]) > 0:
                    new_available_datasets.append(name)
                    new_available_weights.append(available_weights[i])
            if len(new_available_datasets) == len(available_datasets):
                break
            available_datasets = new_available_datasets
            total_w = sum(new_available_weights)
            if total_w > 0:
                available_weights = [w / total_w for w in new_available_weights]
            else:
                available_weights = [1.0 / len(new_available_datasets)] * len(new_available_datasets)

            remaining = num_pids - len(selected_pids)

        if len(selected_pids) < num_pids:
            all_remaining = []
            for name in self.dataset_names:
                all_remaining.extend(available_pids_per_ds[name])
            if len(all_remaining) >= num_pids - len(selected_pids):
                extra = random.sample(all_remaining, num_pids - len(selected_pids))
                selected_pids.extend(extra)
                for pid in extra:
                    for name in self.dataset_names:
                        if pid in available_pids_per_ds[name]:
                            available_pids_per_ds[name].remove(pid)
                            break

        return selected_pids

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

        avai_pids_per_ds = {}
        for name in self.dataset_names:
            avai_pids_per_ds[name] = [pid for pid in self.dataset_pids[name]
                                       if len(batch_idxs_dict[pid]) > 0]

        final_idxs = []
        min_available = min(len(v) for v in avai_pids_per_ds.values()) if avai_pids_per_ds else 0

        while min_available > 0 or any(len(v) >= self.num_pids_per_batch for v in avai_pids_per_ds.values()):
            total_available = sum(len(v) for v in avai_pids_per_ds.values())
            if total_available < self.num_pids_per_batch:
                break

            selected_pids = self._sample_pids_from_datasets(
                self.num_pids_per_batch, avai_pids_per_ds)

            if len(selected_pids) < self.num_pids_per_batch:
                break

            random.shuffle(selected_pids)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    for name in self.dataset_names:
                        if pid in avai_pids_per_ds[name]:
                            avai_pids_per_ds[name].remove(pid)
                            break

            min_available = min(len(v) for v in avai_pids_per_ds.values()) if avai_pids_per_ds else 0

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

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

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

