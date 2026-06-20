import os.path as osp

from .bases import BaseImageDataset
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .cuhk03 import CUHK03
from .viper import VIPeR
from .ilids_vid import iLIDSVID
from .regdb import RegDB
from .prcc import PRCC
from .ltcc import LTCC
from .last import LAST


class CombinedReID(BaseImageDataset):
    """
    CombinedReID: 综合行人重识别数据集

    将多个业内常用的 ReID 数据集汇总到一个 Dataset 中，用于模型训练。
    仅使用各数据集的 RGB 行人图像，合并所有 split（train/query/gallery）的行人数据做模型训练。
    验证集和测试集由用户自行准备（CLIP-REID 的验证测试集）。

    汇总的数据集列表:
        1. Market-1501      - 1,501 identities, 32,217 images, 6 cameras
        2. DukeMTMC-reID    - 1,404 identities, 36,411 images, 8 cameras
        3. CUHK03           - 1,467 identities, 14,097 images, 2 cameras (detected)
        4. VIPeR            - 632 identities, 1,264 images, 2 cameras
        5. iLIDS-VID        - 300 identities, 43,987 images, 2 cameras
        6. RegDB            - 412 identities, 4,120 images, 1 camera (仅visible/RGB)
        7. PRCC             - 921 identities, 33,698 images, 3 cameras
        8. LTCC             - 152 identities, 17,138 images, 12 cameras
        9. LAST             - 10,387 identities, 47,915 images, 4 cameras (仅RGB模态)

    合并策略:
        - 每个子数据集的 pid 全局重新编号，避免不同数据集间的身份冲突
        - 每个子数据集的 camid 全局重新编号，避免不同数据集间的摄像头冲突
        - 所有子数据集的 train/query/gallery 合并为单一训练集
        - query 和 gallery 置空，由用户自行准备验证测试集
        - 若某个子数据集目录不存在，则跳过并打印警告信息
    """

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(CombinedReID, self).__init__()
        self.pid_begin = pid_begin
        self.root = root

        # 定义要汇总的子数据集列表: (名称, 类, 描述)
        self.sub_datasets = [
            ('Market-1501', Market1501, '1,501 identities, 32,217 images, 6 cameras'),
            ('DukeMTMC-reID', DukeMTMCreID, '1,404 identities, 36,411 images, 8 cameras'),
            ('CUHK03', CUHK03, '1,467 identities, 14,097 images, 2 cameras'),
            ('VIPeR', VIPeR, '632 identities, 1,264 images, 2 cameras'),
            ('iLIDS-VID', iLIDSVID, '300 identities, 43,987 images, 2 cameras'),
            ('RegDB', RegDB, '412 identities, 4,120 images, 1 camera (visible only)'),
            ('PRCC', PRCC, '921 identities, 33,698 images, 3 cameras'),
            ('LTCC', LTCC, '152 identities, 17,138 images, 12 cameras'),
            ('LAST', LAST, '10,387 identities, 47,915 images, 4 cameras (RGB only)'),
        ]

        all_train = []
        current_pid_offset = pid_begin
        current_cam_offset = 0
        self.dataset_stats = []  # 记录每个子数据集的统计信息

        for name, cls, desc in self.sub_datasets:
            try:
                sub_dataset = cls(root=root, verbose=False, pid_begin=0)
                # 合并该子数据集的所有 split（train + query + gallery）
                all_images = sub_dataset.train + sub_dataset.query + sub_dataset.gallery

                # 全局重新编号 pid 和 camid，避免不同数据集间冲突
                pid_map = {}
                cam_map = {}
                next_pid = current_pid_offset
                next_cam = current_cam_offset
                remapped = []
                for img_path, pid, camid, trackid in all_images:
                    if pid not in pid_map:
                        pid_map[pid] = next_pid
                        next_pid += 1
                    if camid not in cam_map:
                        cam_map[camid] = next_cam
                        next_cam += 1
                    remapped.append((img_path, pid_map[pid], cam_map[camid], trackid))

                all_train.extend(remapped)
                num_pids = len(pid_map)
                num_imgs = len(remapped)
                num_cams = len(cam_map)
                self.dataset_stats.append((name, num_pids, num_imgs, num_cams, desc))

                if verbose:
                    print("=> {:<15s} loaded: {:6d} images, {:5d} pids, {:3d} cams"
                          .format(name, num_imgs, num_pids, num_cams))

                current_pid_offset = next_pid
                current_cam_offset = next_cam

            except Exception as e:
                if verbose:
                    print("Warning: Could not load {}: {}".format(name, e))

        self.train = all_train
        # query 和 gallery 置空，由用户自行准备 CLIP-REID 验证测试集
        self.query = []
        self.gallery = []

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = 0, 0, 0, 0
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = 0, 0, 0, 0

        if verbose:
            self.print_combined_statistics()

    def print_combined_statistics(self):
        print("=" * 70)
        print("CombinedReID Dataset Statistics")
        print("=" * 70)
        print("  {:<15s} | {:>6s} | {:>7s} | {:>5s} | {}"
              .format("Dataset", "# pids", "# imgs", "# cam", "Description"))
        print("  " + "-" * 65)
        for name, num_pids, num_imgs, num_cams, desc in self.dataset_stats:
            print("  {:<15s} | {:6d} | {:7d} | {:5d} | {}"
                  .format(name, num_pids, num_imgs, num_cams, desc))
        print("  " + "-" * 65)
        print("  {:<15s} | {:6d} | {:7d} | {:5d} |"
              .format("TOTAL", self.num_train_pids, self.num_train_imgs, self.num_train_cams))
        print("=" * 70)
        print("  Note: query and gallery are empty. Please prepare your own")
        print("        validation/test set for CLIP-REID evaluation.")
        print("=" * 70)
