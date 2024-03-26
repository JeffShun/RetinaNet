import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import ResNet
from custom.model.neck import PyramidFeatures
from custom.model.anchor import Anchors
from custom.model.loss import Focal_Loss
from custom.model.head import Detection_Head
from custom.model.decoder import Decoder
from custom.model.network import Detection_Network

class network_cfg:

    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

    # img
    img_size = (320, 320)
    lable_map = {0:"knee"}
    
    # network
    fpn_sizes = [512, 1024, 2048]
    pyramid_levels = [3, 4, 5]
    box_scale_factor = [0.1, 0.1, 0.2, 0.2]

    network = Detection_Network(
        backbone = ResNet(
            in_channel=1, 
            block_name="Bottleneck",
            layers=[3, 4, 6, 3]
            ),
        neck = PyramidFeatures(
            C3_size=fpn_sizes[0], 
            C4_size=fpn_sizes[1],
            C5_size=fpn_sizes[2], 
            feature_size=256
        ),
        anchor_generator=Anchors(
            pyramid_levels=pyramid_levels,
            strides=[2**x for x in pyramid_levels],
            sizes=[135, 180, 245],
            ratios=[0.5, 1, 2],
            scales=[2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        ),
        decoder=Decoder(
            image_h=img_size[0],
            image_w=img_size[1],
            scale_factor=box_scale_factor,
            top_n=1000,
            min_score_threshold=0.5,
            nms_threshold=0.1,
            max_detection_num=1,
        ),
        head=Detection_Head(
            num_features_in=256,
            num_classes=1,
            num_anchors=9,
            feature_size=256
        ),
    apply_sync_batchnorm=False
    )


    # loss function
    train_loss_f = Focal_Loss(alpha=0.75, gamma=2.0, pos_thresh=0.4, neg_thresh=0.2, scale_factor=box_scale_factor)
    valid_loss_f = Focal_Loss(alpha=0.75, gamma=2.0, pos_thresh=0.4, neg_thresh=0.2, scale_factor=box_scale_factor)

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            resize(img_size),
            normlize(win_clip=None),
            random_flip(axis=1, prob=0.5),
            random_flip(axis=2, prob=0.5),
            random_rotate90(prob=0.5),
            label_alignment(max_box_num=1, pad_val=-1)
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            resize(img_size),
            normlize(win_clip=None),
            random_flip(axis=1, prob=0.5),
            random_flip(axis=2, prob=0.5),
            random_rotate90(prob=0.5),
            label_alignment(max_box_num=1, pad_val=-1)
            ])
        )
    
    # train dataloader
    batchsize = 2
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [40,80,120]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 150
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/Resnet50"
    checkpoints_dir = work_dir + '/checkpoints/Resnet50'
    load_from = work_dir + '/checkpoints/Resnet50/150.pth'
