from ultralytics import YOLO
'''See: https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation'''

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# View existing model confidence
print("Model.conf: {model.conf}")

# Set new model confidence
model.conf = .4

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category