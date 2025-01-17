_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
