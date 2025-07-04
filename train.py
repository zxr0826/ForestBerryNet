import warnings, os
warnings.filterwarnings('ignore')
from ForestBerryNet import YOLO
if __name__ == '__main__':
    model = YOLO(f'/home/waas/ForestBerry-copy/ForestBerryNet/cfg/models/11/ForestBerryNet.yaml')
    model.train(data='/home/waas/data/wildB/data.yaml',
                cache=False,
                imgsz=640,
                epochs=600,
                batch=32,
                close_mosaic=0,
                workers=4, 
                optimizer='SGD',
                project='runs/train',
                name='FB',
                )