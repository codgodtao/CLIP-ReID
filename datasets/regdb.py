import glob
import os.path as osp

from .bases import BaseImageDataset


class RegDB(BaseImageDataset):
    """
    RegDB (Person Re-identification by Cross-Modal Visible-Thermal)
    Reference:
        Nguyen et al. "Person Re-identification via Latent Maximum Spacing Analysis."
        IEEE Transactions on Image Processing 2017.
    URL: http://dm.dongguk.edu/link.html

    Dataset statistics:
    # identities: 412 (206 train + 206 test, per trial)
    # images: 4,120 visible (RGB) + 4,120 thermal (本加载器仅使用 visible/RGB 图像)
    # cameras: 2 (visible + thermal)，本加载器仅使用 visible 摄像头
    # 特性: 可见光-红外跨模态行人重识别数据集。每个身份在可见光和热红外摄像头下各有10张图像。
           根据需求，本加载器仅加载 visible（RGB彩色）行人图像，忽略 thermal 红外图像。
           数据集提供10组随机划分（trials），每组206训练/206测试。

    Directory structure:
        RegDB/
            visible/
                <pid>/
                    *.jpg
            thermal/
                <pid>/
                    *.jpg
            splits/
                test_1.txt
                ...
    """
    dataset_dir = 'RegDB'
    # 仅使用 visible (RGB) 图像，忽略 thermal 红外图像
    modality = 'visible'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(RegDB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.visible_dir = osp.join(self.dataset_dir, self.visible)

        self.pid_begin = pid_begin
        self._check_before_run()

        # 仅加载 visible (RGB) 图像，合并所有身份用于综合训练
        all_data = self._process_dir()

        if verbose:
            print("=> RegDB loaded (visible/RGB only)")
            self.print_dataset_statistics(all_data, [], [])

        self.train = all_data
        self.query = []
        self.gallery = []

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    @property
    def visible(self):
        return self.modality

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.visible_dir):
            raise RuntimeError("'{}' is not available".format(self.visible_dir))

    def _process_dir(self):
        # RegDB visible 目录结构: visible/<pid>/*.jpg
        # 每个身份有10张可见光图像，统一使用 camid=0
        pid_container = set()
        for pid_dir in glob.glob(osp.join(self.visible_dir, '*')):
            pid = int(osp.basename(pid_dir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for pid_dir in sorted(glob.glob(osp.join(self.visible_dir, '*'))):
            pid = int(osp.basename(pid_dir))
            pid = pid2label[pid]
            for img_path in sorted(glob.glob(osp.join(pid_dir, '*.jpg'))):
                dataset.append((img_path, self.pid_begin + pid, 0, 0))
        return dataset
