from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS


@MODELS.register_module()
class SwinUNet(BaseModule):
    def __init__(self):
        return
