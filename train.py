import warnings, os
warnings.filterwarnings('ignore')
from ForestBerryNet import YOLO
if __name__ == '__main__':
    model = YOLO(f'/path/your/yaml/')
    model.train(data='/path/dataset/yaml/',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, 
                optimizer='SGD',
                project='runs/train',
                name='FB',
                )
