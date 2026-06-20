import glob
import os.path as osp

from .bases import BaseImageDataset


class VIPeR(BaseImageDataset):
    """
    VIPeR
    Reference:
        Gray et al. "Evaluating appearance models for recognition, reacquisition, and tracking."
        IEEE PETS Workshop 2007.
    URL: https://vision.soe.ucsc.edu/node/178

    Dataset statistics:
    # identities: 632
    # images: 1,264 (每个身份2张图像，分别来自2个摄像头)
    # cameras: 2 (cam_a, cam_b)
    # 特性: 早期经典行人重识别数据集，规模较小。每个身份在两个摄像头下各有一张图像。
           图像为RGB彩色图像(.bmp格式)。无预定义训练/测试划分，通常采用随机划分并多次实验取平均。

    Directory structure:
        VIPeR/
            cam_a/
                <pid>.bmp
            cam_b/
                <pid>.bmp
    """
    dataset_dir = 'VIPeR'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(VIPeR, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cam_a_dir = osp.join(self.dataset_dir, 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'cam_b')

        self.pid_begin = pid_begin
        self._check_before_run()

        # VIPeR 没有预定义的 train/test 划分，加载所有图像用于综合训练
        all_data = self._process_dir()

        if verbose:
            print("=> VIPeR loaded")
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

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.cam_a_dir):
            raise RuntimeError("'{}' is not available".format(self.cam_a_dir))
        if not osp.exists(self.cam_b_dir):
            raise RuntimeError("'{}' is not available".format(self.cam_b_dir))

    def _process_dir(self):
        # VIPeR 目录结构: cam_a/<pid>.bmp 和 cam_b/<pid>.bmp
        # 同一 pid 在 cam_a 和 cam_b 中各有一张图像
        dataset = []
        pid_container = set()

        # 收集所有 pid
        for cam_dir, camid in [(self.cam_a_dir, 0), (self.cam_b_dir, 1)]:
            for img_path in glob.glob(osp.join(cam_dir, '*.bmp')):
                pid = int(osp.splitext(osp.basename(img_path))[0])
                pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        # 处理所有图像
        for cam_dir, camid in [(self.cam_a_dir, 0), (self.cam_b_dir, 1)]:
            for img_path in sorted(glob.glob(osp.join(cam_dir, '*.bmp'))):
                pid = int(osp.splitext(osp.basename(img_path))[0])
                pid = pid2label[pid]
                dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
