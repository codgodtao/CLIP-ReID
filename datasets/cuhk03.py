import glob
import os.path as osp

from .bases import BaseImageDataset


class CUHK03(BaseImageDataset):
    """
    CUHK03
    Reference:
        Li et al. "DeepReID: Deep Filter Pairing Neural Network for Person Re-identification." CVPR 2014.
        New split protocol: Li et al. "Deep Learning for Person Re-identification: A Survey and Outlook." arXiv 2016.
    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html

    Dataset statistics:
    # identities: 1,467 (767 train + 700 test, new protocol)
    # images: 14,097 (detected version) / 13,564 (labeled version)
    # cameras: 2 (camera pairs from 5 disjoint pairs of cameras)
    # 特性: 早期大规模行人重识别数据集，包含人工标注(labeled)和DPM检测(detected)两个版本。
           采用 Li et al. 提出的新划分协议(new protocol)。所有图像为RGB彩色图像。

    Directory structure:
        cuhk03/
            detected/
                train/
                    <pid>/
                        *.png
                test/
                    <pid>/
                        *.png
            labeled/
                train/
                    <pid>/
                        *.png
                test/
                    <pid>/
                        *.png
    """
    dataset_dir = 'cuhk03'
    # 默认使用 detected 版本（DPM检测边界框），可选 'labeled' 使用人工标注版本
    version = 'detected'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(CUHK03, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.version_dir = osp.join(self.dataset_dir, self.version)
        self.train_dir = osp.join(self.version_dir, 'train')
        self.test_dir = osp.join(self.version_dir, 'test')

        self.pid_begin = pid_begin
        self._check_before_run()

        # 加载训练集和测试集，合并所有 split 用于综合训练
        train = self._process_dir(self.train_dir, relabel=True)
        test = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> CUHK03 loaded (version={})".format(self.version))
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
        if not osp.exists(self.version_dir):
            raise RuntimeError("'{}' is not available".format(self.version_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        # CUHK03 目录结构: dir_path/<pid>/*.png
        # 文件名格式通常为 <camid>_<sequence>_<frame>.png，这里从文件名提取摄像头ID
        pid_container = set()
        for pid_dir in glob.glob(osp.join(dir_path, '*')):
            pid = int(osp.basename(pid_dir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for pid_dir in sorted(glob.glob(osp.join(dir_path, '*'))):
            pid = int(osp.basename(pid_dir))
            if relabel:
                pid = pid2label[pid]
            for img_path in sorted(glob.glob(osp.join(pid_dir, '*.png'))):
                # CUHK03 文件名格式: <camid>_<...>.png，摄像头ID从文件名首字符提取
                basename = osp.basename(img_path)
                camid = int(basename.split('_')[0]) - 1  # camid 从0开始
                dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
