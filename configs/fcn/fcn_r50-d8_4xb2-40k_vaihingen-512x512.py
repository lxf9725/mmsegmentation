_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/vaihingen.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(align_corners=True,
                     num_classes=6),
    auxiliary_head=dict(align_corners=True,
                        num_classes=6))
