import os
from pathlib import Path
from easydict import EasyDict as edict

root_pth = str(Path(__file__).parent)


c = edict()

# Meta settings
c.meta = edict()
c.meta.root_dir = root_pth
c.meta.round_number = 8
c.meta.debug = False
c.meta.device = "cuda"
c.meta.test_device = "cpu"

# Hyperparameters
c.hps = edict()
c.hps.batch_size = 4
c.hps.test_batch_size = 1  # must be set to 1 to evaluate metric
c.hps.img_size = 640
c.hps.no_epochs = 100
c.hps.warmup_epochs = 10
c.hps.no_classes = 2
c.hps.log_iter = 50

# Learning rates
c.lrs = edict()
c.lrs.mode = "reduce"  # reduce / poly
c.lrs.warmup_iters = 10
c.lrs.factor = 0.2
c.lrs.patience = 4

# Augmentation settings
c.augmentation = edict()

# Callbacks settings
c.callbacks = edict()

# Data settings
c.data = edict()
c.data.icdar2015 = edict()
c.data.icdar2015.train_dir = Path(root_pth).joinpath(
    "Nepali_Text_Detection_Dataset/train/images/"
)
c.data.icdar2015.test_dir = Path(root_pth).joinpath(
    "Nepali_Text_Detection_Dataset/test/images/"
)
c.data.icdar2015.train_gt_dir = Path(root_pth).joinpath(
    "Nepali_Text_Detection_Dataset/train/gt"
)
c.data.icdar2015.test_gt_dir = Path(root_pth).joinpath(
    "Nepali_Text_Detection_Dataset/test/gt/"
)
c.data.icdar2015.ignore_tags = ["###"]


# Dataset settings
c.dataset = edict()
c.dataset.name = "icdar2015"
c.dataset.return_dict = True

# Logging settings
c.logging = edict()
c.logging.logger_file = "train.log"

# Loss settings
c.loss = edict()

# Model settings
c.model = edict()
c.model.finetune_cp_path = ""
c.model.best_cp_path = "models/nepali/nepali_td_best.pth"
c.model.last_cp_path = "models/nepali/nepali_td.pth"
c.model.best_hmean_cp_path = "models/nepali/nepali_td_best_hmean.pth"

# Optimizer settings
c.optimizer = edict()
c.optimizer.type = "adam"
c.optimizer.lr = 0.005
c.optimizer.lr_finetune = 0.001
c.optimizer.weight_decay = 0.0
c.optimizer.reduction = "mean"
c.optimizer.alpha = 1
c.optimizer.beta = 10
c.optimizer.negative_ratio = 3
c.optimizer.amsgrad = False

# Metric settings
c.metric = edict()
c.metric.thred_text_score = 0.25
c.metric.prob_threshold = 0.50
c.metric.unclip_ratio = 1.50
c.metric.is_output_polygon = True

# Private settings
c.private = edict()

# Scheduler settings
c.scheduler = edict()

# Trainer settings
c.trainer = edict()

# Training settings
c.training = edict()

cfg = c
