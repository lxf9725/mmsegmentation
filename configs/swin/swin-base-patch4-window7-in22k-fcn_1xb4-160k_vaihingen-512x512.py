_base_ = [
    './swin-base-patch4-window7-in1k-fcn_1xb4-160k_potsdam-512x512.py'
]
checkpoint_file = 'pretrain/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)))
