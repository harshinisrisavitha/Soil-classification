from ultralytics import YOLO


model = YOLO("yolov8n-cls.pt")


model.train(data="YOLO/soil_dataset", epochs=20, imgsz=224)


model=YOLO('runs/classify/train3/weights/best.pt')

metrics = model.val()
print("Validation Metrics:", metrics)


results = model.predict(source="YOLO/gravel.png",save=True,save_txt=True,show=True)
print(results)


