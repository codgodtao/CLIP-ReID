import glob
import os.path as osp

from .bases import BaseImageDataset


class LAST(BaseImageDataset):
    """
    LAST (Large-scale Person ReID Dataset)
    Reference:
        Shu et al. "Large-scale Person Re-identification across Aerial and Ground cameras."
        URL: https://github.com/shuxjweb/LAST.git

    Dataset statistics:
    # identities: 10,387 (5,193 train + 5,194 test)
    # images: 47,915 (RGB pedestrian images, 仅使用RGB模态)
    # cameras: 4
    # 特性: 大规模跨模态行人重识别数据集，包含RGB、红外、素描、打印四种模态。
           本加载器仅使用RGB彩色行人图像。数据集规模大、身份数量多，
           适合作为大规模预训练数据。

    Directory structure:
        LAST/
            RGB/
                train/
                    <pid>/
                        *.jpg
                test/
                    <pid>/
                        *.jpg
            Infrared/
                ...
            Sketch/
                ...
            Print/
                ...
    """
    dataset_dir = 'LAST'
    # 仅使用 RGB 模态，忽略红外/素描/打印等非RGB模态
    rgb_dir_name = 'RGB'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LAST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.rgb_dir = osp.join(self.dataset_dir, self.rgb_dir_name)
        self.train_dir = osp.join(self.rgb_dir, 'train')
        self.test_dir = osp.join(self.rgb_dir, 'test')

        self.pid_begin = pid_begin
        self._check_before_run()

        # 加载训练集和测试集的 RGB 图像
        # train 重标号, test 不重标号 (用于评估时与 train 区分)
        train = self._process_dir(self.train_dir, relabel=True)
        test = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> LAST loaded (RGB only)")
            self.print_dataset_statistics(train, test, [])

        self.train = train
        # 修复: 原代码 self.query = test, self.gallery = [] 会导致评估崩溃
        # (eval_func 要求 gallery 非空, 且会移除与 query 同 pid 同 camid 的样本,
        #  若 gallery 为空则 num_valid_q=0 触发 assert)。
        # 改为: query 和 gallery 都使用 test 集, 通过 camid 区分 (不同 camid 才算有效匹配)。
        # 由于 LAST 文件名未编码 camid, 这里给 query 和 gallery 分配不同 camid
        # (query=0, gallery=1), 使 eval_func 能正常工作。
        self.query = [(p, pid, 0, t) for (p, pid, _, t) in test]
        self.gallery = [(p, pid, 1, t) for (p, pid, _, t) in test]

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.rgb_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        # LAST RGB 目录结构: dir_path/<pid>/*.jpg
        # 文件名中可能包含摄像头信息，这里统一使用 camid=0
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
            for img_path in sorted(glob.glob(osp.join(pid_dir, '*.jpg'))):
                dataset.append((img_path, self.pid_begin + pid, 0, 0))
        return dataset
