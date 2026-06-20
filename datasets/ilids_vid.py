import glob
import os.path as osp

from .bases import BaseImageDataset


class iLIDSVID(BaseImageDataset):
    """
    iLIDS-VID
    Reference:
        Wang et al. "Person Re-Identification by Video Ranking." ECCV 2014.
    URL: http://www.eecs.qmul.ac.uk/~xiatian/person_re_id.html

    Dataset statistics:
    # identities: 300 (150 train + 150 test)
    # images: 43,987 frames (平均每个视频序列73帧)
    # cameras: 2 (cam1, cam2)
    # 特性: 基于视频的行人重识别数据集，每个身份在两个摄像头下各有一段视频序列。
           采集自机场到达大厅监控视频。本加载器将所有视频帧作为独立图像样本使用。
           所有图像为RGB彩色图像(.png格式)。

    Directory structure:
        i-LIDS-VID/
            train/
                <pid>/
                    cam1/
                        *.png
                    cam2/
                        *.png
            test/
                <pid>/
                    cam1/
                        *.png
                    cam2/
                        *.png
    """
    dataset_dir = 'i-LIDS-VID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(iLIDSVID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self.pid_begin = pid_begin
        self._check_before_run()

        # 加载训练集和测试集的所有视频帧，合并所有 split 用于综合训练
        train = self._process_dir(self.train_dir, relabel=True)
        test = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> iLIDS-VID loaded")
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
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        # iLIDS-VID 目录结构: dir_path/<pid>/cam{1,2}/*.png
        # 每个身份在 cam1 和 cam2 下各有一段视频序列，将每帧作为独立样本
        cam_map = {'cam1': 0, 'cam2': 1}
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
            for cam_name, camid in cam_map.items():
                cam_dir = osp.join(pid_dir, cam_name)
                if not osp.exists(cam_dir):
                    continue
                for img_path in sorted(glob.glob(osp.join(cam_dir, '*.png'))):
                    dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
