from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_lsj import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "/mnt/cache/liuyuan/research/mae/pretrain_s2/mae_retrain/checkpoint-299.pth"
train.output_dir = "work_dir/mae_det/mae_300e"

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# ceph
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'datasets/': 's3://openmmlab/datasets/detection/',
        '.datasets/': 's3://openmmlab/datasets/detection/'
    }))

# If you don’t need ceph, you can directly comment the following code or
# set file_client_args to None
dataloader.train.mapper.file_client_args = file_client_args
dataloader.test.mapper.file_client_args = file_client_args

dataloader.train.num_workers = 8
dataloader.test.num_workers = 8
