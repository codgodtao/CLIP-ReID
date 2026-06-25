from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import math
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None
import torch
import pickle


class BalancedDatasetSampler_DDP(Sampler):
    """
    Balanced Dataset Sampler (DDP version): 分布式版本的数据集平衡采样器

    支持的平衡模式 (balance_mode):
    - 'uniform': 每个数据集被选中的概率相同（均匀分布）
    - 'proportional': 按数据集身份数量比例采样
    - 'square_root': 按数据集身份数量的平方根比例采样

    Args:
    - data_source (list): list of (img_path, pid, camid, trackid)
    - batch_size (int): 全局 batch size
    - num_instances (int): number of instances per identity in a batch
    - dataset_pid_ranges (dict): {dataset_name: (pid_start, pid_end)}
    - balance_mode (str): 平衡模式
    - dataset_weights (dict or None): 自定义权重
    """

    def __init__(self, data_source, batch_size, num_instances,
                 dataset_pid_ranges=None, balance_mode='uniform',
                 dataset_weights=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
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

        self.rank = dist.get_rank()
        self.length //= self.world_size

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

        while len(selected_pids) < num_pids and len(available_datasets) > 0:
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
                if len(selected_pids) >= num_pids:
                    break
                n_select = min(counts_per_ds.get(name, 0), len(available_pids_per_ds[name]))
                n_select = min(n_select, num_pids - len(selected_pids))
                if n_select > 0:
                    chosen = np.random.choice(available_pids_per_ds[name], n_select, replace=False).tolist()
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

        if len(selected_pids) < num_pids:
            all_remaining = []
            for name in self.dataset_names:
                all_remaining.extend(available_pids_per_ds[name])
            if len(all_remaining) >= num_pids - len(selected_pids):
                extra = np.random.choice(all_remaining, num_pids - len(selected_pids), replace=False).tolist()
                selected_pids.extend(extra)
                for pid in extra:
                    for name in self.dataset_names:
                        if pid in available_pids_per_ds[name]:
                            available_pids_per_ds[name].remove(pid)
                            break

        return selected_pids

    def __iter__(self):
        seed = shared_random_seed()
        np.random.seed(seed)
        random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.mini_batch_size * self.rank + self.mini_batch_size * i,
                          min(self.mini_batch_size * self.rank + self.mini_batch_size * (i+1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs

    def sample_list(self):
        avai_pids_per_ds = {}
        batch_idxs_dict = {}
        for name in self.dataset_names:
            avai_pids_per_ds[name] = copy.deepcopy(self.dataset_pids[name])

        batch_indices = []
        total_pids = sum(len(v) for v in avai_pids_per_ds.values())

        while total_pids >= self.num_pids_per_batch:
            selected_pids = self._sample_pids_from_datasets(
                self.num_pids_per_batch, avai_pids_per_ds)

            if len(selected_pids) < self.num_pids_per_batch:
                break

            np.random.shuffle(selected_pids)
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    del batch_idxs_dict[pid]
                    for name in self.dataset_names:
                        if pid in avai_pids_per_ds[name]:
                            avai_pids_per_ds[name].remove(pid)
                            break

            total_pids = sum(len(v) for v in avai_pids_per_ds.values())

        return batch_indices

    def __len__(self):
        return self.length


def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

class RandomIdentitySampler_DDP(Sampler):
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
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.index_dic = defaultdict(list)

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

        self.rank = dist.get_rank()
        #self.world_size = dist.get_world_size()
        self.length //= self.world_size

    def __iter__(self):
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        #final_idxs = final_idxs[self.rank * length:(self.rank + 1) * length]
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)


    def __fetch_current_node_idxs(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.mini_batch_size * self.rank + self.mini_batch_size * i, min(self.mini_batch_size * self.rank + self.mini_batch_size * (i+1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs


    def sample_list(self):
        #np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length
