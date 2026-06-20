import os.path as osp

from .bases import BaseImageDataset


class LTCC(BaseImageDataset):
    """
    LTCC (Long-Term Clothes-Changing) ReID Dataset
    Reference:
        Qian et al. "Long-Term Cloth-Changing Person Re-identification." ACCV 2020.
    URL: https://naiq.github.io/LTCC_Perosn_ReID.html

    Dataset statistics:
    # identities: 152 (91 train + 61 test)
    # images: 17,138 (RGB pedestrian images)
    # cameras: 12
    # 特性: 长期换衣行人重识别数据集。包含衣物变化场景，共478套衣服。
           其中7,625张图像涉及衣物变化。所有图像为RGB彩色图像。

    Directory structure:
        LTCC_ReID/
            images/
                <pid>_<clothes_id>_<camid>.jpg
            train_list.txt   (format: img_name pid clothes_id camid)
            test_list.txt
            query_list.txt
    """
    dataset_dir = 'LTCC_ReID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LTCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_list.txt')
        self.test_list = osp.join(self.dataset_dir, 'test_list.txt')
        self.query_list = osp.join(self.dataset_dir, 'query_list.txt')

        self.pid_begin = pid_begin
        self._check_before_run()

        # 加载训练集、查询集和图库集，合并所有 split 用于综合训练
        train = self._process_list(self.train_list, relabel=True)
        query = self._process_list(self.query_list, relabel=False)
        gallery = self._process_list(self.test_list, relabel=False)

        if verbose:
            print("=> LTCC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not osp.exists(self.test_list):
            raise RuntimeError("'{}' is not available".format(self.test_list))

    def _process_list(self, list_path, relabel=False):
        # LTCC 列表文件格式: img_name pid clothes_id camid
        with open(list_path, 'r') as f:
            lines = f.readlines()

        pid_container = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            _, pid, _, _ = line.split(' ')
            pid_container.add(int(pid))
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            img_name, pid, clothes_id, camid = line.split(' ')
            pid = int(pid)
            camid = int(camid)
            if relabel:
                pid = pid2label[pid]
            img_path = osp.join(self.img_dir, img_name)
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 0))
        return dataset
