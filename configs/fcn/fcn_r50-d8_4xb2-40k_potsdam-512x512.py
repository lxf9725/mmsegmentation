_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/potsdam.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
