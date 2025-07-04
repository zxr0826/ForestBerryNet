try:
    import swattention
    from ForestBerryNet.nn.backbone.TransNeXt.TransNext_cuda import *
except ImportError as e:
    from ForestBerryNet.nn.backbone.TransNeXt.TransNext_native import *
    pass

