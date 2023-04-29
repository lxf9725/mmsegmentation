_base_ = [
    '../_base_/models/swin_resnet.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
res_checkpoint_file = ''
swin_checkpoint_file = 'pretrain/swin_base_patch4_window7_224.pth'
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=1024, num_classes=6))
