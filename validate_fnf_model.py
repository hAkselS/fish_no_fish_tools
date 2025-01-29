from ultralytics import YOLO
'''See: https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation'''

class Args():
    def __init__(self):
        self.model_path = "/home/gpu_enjoyer/fish_no_fish_tools/models/accepted/fnf_v1/train3(fnf_candidate_2)/weights/best(used in fnf on MM).pt"
        self.dataset_yaml = "/home/gpu_enjoyer/fish_no_fish_tools/data/fish_dataset.yaml"
        self.confidence = .001 # 0.001 is the default value for the confidence in validations

# Instantiate a class to hold variables
args = Args() 

# Load a model
model = YOLO(args.model_path)  # load a custom model



# Validate the model
metrics = model.val(data=args.dataset_yaml, conf = args.confidence)  
print(f"map50-95: {metrics.box.map}")  # map50-95
print(f"map50: {metrics.box.map50}")  # map50
# print(f"map75: {metrics.box.map75}")  # map75
print(f"precision: {metrics.box.mp}")
print(f"recall: {metrics.box.mp}")
#metrics.box.maps  # a list contains map50-95 of each category