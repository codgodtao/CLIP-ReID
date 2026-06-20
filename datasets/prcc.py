import glob
import os.path as osp

from .bases import BaseImageDataset


class PRCC(BaseImageDataset):
    """
    PRCC (Person Re-identification by Contour and Color)
    Reference:
        Yang et al. "Spatial and Temporal Representations for Large-Scale Person Re-Identification." IJCV 2023.
        Original: Yang et al. "Beyond Scalar Neuron: Adopting Vector-Neuron Capsules for Long-Term Person
        Re-Identification." IEEE TCSVT 2019.
    URL: https://www.isee-ai.cn/~yangqize/prcc.html

    Dataset statistics:
    # identities: 921 (669 train + 252 test)
    # images: 33,698 (RGB pedestrian images)
    # cameras: 3 (cam A, cam B, cam C)
    # 特性: 换衣行人重识别数据集。摄像头A和B拍摄同一套衣服，摄像头C拍摄不同衣服。
           适合用于研究衣物变化条件下的行人重识别。所有图像为RGB彩色图像。

    Directory structure:
        prcc/
            dataset/
                rgb/
                    train/
                        <pid>/
                            A/*.jpg
                            B/*.jpg
                    test/
                        <pid>/
                            A/*.jpg
                            B/*.jpg
                            C/*.jpg
    """
    dataset_dir = 'prcc'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # PRCC 数据集的 RGB 图像目录
        self.data_dir = osp.join(self.dataset_dir, 'dataset', 'rgb')
        self.train_dir = osp.join(self.data_dir, 'train')
        self.test_dir = osp.join(self.data_dir, 'test')

        self.pid_begin = pid_begin
        self._check_before_run()

        # 加载训练集和测试集，合并所有 split 用于综合训练
        train = self._process_dir(self.train_dir, relabel=True)
        test = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> PRCC loaded")
            self.print_dataset_statistics(train, test, [])

        self.train = train
        self.query = test
        self.gallery = []

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        # PRCC 目录结构: dir_path/<pid>/<cam_letter>/*.jpg
        # 摄像头映射: A->0, B->1, C->2
        cam_map = {'A': 0, 'B': 1, 'C': 2}
        pid_container = set()
        # 先收集所有 pid
        for pid_dir in glob.glob(osp.join(dir_path, '*')):
            pid = int(osp.basename(pid_dir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for pid_dir in sorted(glob.glob(osp.join(dir_path, '*'))):
            pid = int(osp.basename(pid_dir))
            if relabel:
                pid = pid2label[pid]
            for cam_letter in ['A', 'B', 'C']:
                cam_dir = osp.join(pid_dir, cam_letter)
                if not osp.exists(cam_dir):
                    continue
                camid = cam_map[cam_letter]
                for img_path in sorted(glob.glob(osp.join(cam_dir, '*.jpg'))):
                    dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
