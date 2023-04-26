_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/potsdam_256.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6))
