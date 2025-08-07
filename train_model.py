from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data/aerial-pool/data.yaml", epochs=50, imgsz=640)


# model = YOLO("runs/detect/train/weights/best.pt")

# results = model.predict(source="data/aerial-pool/test/images", save=True, conf=0.5)

# metrics = model.val(data="data/aerial-pool/data.yaml", split="test")



model.train(
    data='config.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    freeze=7,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1
)