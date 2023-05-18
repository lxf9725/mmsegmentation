_base_ = [
    './swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_vaihingen-512x512.py'
]
checkpoint_file = 'pretrain/swin_base_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=6),
    auxiliary_head=dict(in_channels=512, num_classes=6))