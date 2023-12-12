_base_ = ['../../../../_base_/datasets/interhand_twohands.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW',
                 lr=5e-4,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'text_encoder': dict(lr_mult=0.0),
                                                 'backbone': dict(lr_mult=0.1),
                                                 'norm': dict(decay_mult=0.)})
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=42,
    dataset_joints=42,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
    ])

# model settings
model = dict(
    type='CLAMP',
    clip_pretrained='pretrained/ViT-B-16.pt',
    context_length=11,
    text_dim=512,
    score_concat_index=3,
    visual_dim=512,
    CL_ratio=0.0005,
    class_names=['right thumb tip', 'right thumb proximal joint', 'right thumb intermediate joint', 'right thumb base joint',
                'right index tip', 'right index proximal joint', 'right index intermediate joint', 'right index base joint',
                'right middle tip', 'right middle proximal joint', 'right middle intermediate joint', 'right middle base joint',
                'right ring tip', 'right ring proximal joint', 'right ring intermediate joint', 'right ring base joint',
                'right little tip', 'right little proximal joint', 'right little intermediate joint', 'right little base joint',
                'right wrist', 'left thumb tip', 'left thumb proximal joint', 'left thumb intermediate joint', 'left thumb base joint',
                'left index tip', 'left index proximal joint', 'left index intermediate joint', 'left index base joint',
                'left middle tip', 'left middle proximal joint', 'left middle intermediate joint', 'left middle base joint',
                'left ring tip', 'left ring proximal joint', 'left ring intermediate joint', 'left ring base joint',
                'left little tip', 'left little proximal joint', 'left little intermediate joint', 'left little base joint',
                'left wrist'],
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pretrained='pretrained/ViT-B-16.pt',
        style='pytorch'),
    prompt_encoder=dict(
        type='PromptEncoderWithoutPositionemb',
        prompt_num=42,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=1,
        embed_dim=512,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    backbone=dict(
        type='CLIPVisionTransformer',
        debug=False,
        use_fpn=False,
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.4,
        layers=12,
        input_resolution=256,
        style='pytorch'),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        in_channels=810, #768+42
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=1.0)),
    upconv_head=dict( #for score map only
        type='TopdownHeatmapSimpleHead',
        num_deconv_layers=2,
        num_deconv_filters=(42, 42),
        num_deconv_kernels=(4, 4),
        in_channels=42,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=2.0)),
    identity_head=dict(
        type='IdentityHead',
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=2.0)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, downtarget=True, downsize=16),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/hands'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    persistent_workers=True,
    pin_memory=False,
    train=dict(
        type='AnimalAP10KDataset',
        ann_file=f'{data_root}/annotations/assemblyhands_train_ego_data_v1-1_twohands.json',
        img_prefix=f'{data_root}/data/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='AnimalAP10KDataset',
        ann_file=f'{data_root}/annotations/assemblyhands_val_ego_data_v1-1_twohands.json',
        img_prefix=f'{data_root}/data/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='AnimalAP10KDataset',
        ann_file=f'{data_root}/annotations/assemblyhands_test_ego_data_v1-1_twohands.json',
        img_prefix=f'{data_root}/data/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
