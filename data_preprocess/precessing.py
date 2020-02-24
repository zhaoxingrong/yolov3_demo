class Dateset(object):
    def __init__(self, dataset_type):
        # 根据dataset_type的值，选择训练/测试的参数
        # 数据注释文件的路径，此处为"./data/dataset/voc_test.txt" 或 "./data/dataset/voc_train.txt"
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        # 数据输入图像的大小，为了增加网络的鲁棒性，使用了随机[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        # 中任意一种大小，注意，该处必须为32的倍数
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        # 数据增强
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG
        # 训练数据输入大小
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        # 3中下采样方式，为[8, 16, 32]
        self.strides = np.array(cfg.YOLO.STRIDES)
        # 训练数据的类别，使用VOC数据共20中，来自"./data/classes/voc.names"
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        # 种类的数目，针对VOC为20
        self.num_classes = len(self.classes)
        # 来自于"./data/anchors/basline_anchors.txt"，该文件的生成于docs/Box-Clustering.ipynb
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        # 对每个gred(网格)预测几个box，该处为3
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # 每一下采样的最大Bounding box数量
        self.max_bbox_per_scale = 150
        # 根据dataset_type的类型,读取"./data/classes/voc_train.txt"或"./data/classes/voc_test.txt"中的内容
        self.annotations = self.load_annotations(dataset_type)
        # 计算训练样本的总数目
        self.num_samples = len(self.annotations)
        # 计算需要多少个mini_batchs才能完成一个EPOCHS
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        # 当batch_count达到num_batchs代表训练了一个EPOCHS
        self.batch_count = 0