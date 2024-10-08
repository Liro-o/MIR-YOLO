from ultralytics import YOLO
import warnings


warnings.filterwarnings('ignore')
model = YOLO(r"D:/PythonSpace/MIR-YOLO/ultralytics/cfg/models/v8/yolov8-twoCSP-MogaNet-ELAN-ViL.yaml")  # 初始化模型
model.train(data=r"D:/PythonSpace/MIR-YOLO/ultralytics/cfg/datasets/mydata.yaml",
            batch=2,
            epochs=200,
            project='runs/detect/train=liro', name='MIR-yolo',
            amp=False,
            workers=0,
            optimizer='AdamW',  # Optimizer
            # cos_lr=True,  # Cosine LR Scheduler
            lr0=0.001,
            device='0'
            )  # 训练

# ############## 这是val和predict的代码 ##############
# model = YOLO(r"F:/lic_files/MyNet/runs/detect/train=2/yolov8-early28/weights/best.pt")
# # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=1, save_json=True, save_txt=False)  # 验证
# model.predict(source=r"F:/lic_files/MyNet/datasets/Dual_Modity/images/test", save=True)  #   检测
