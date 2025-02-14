"""
File:   train_fnf_model.py

Spec:   Use this script to train yolo 11 nano, medium or larger models. 
        All arguements are stored in the Args class such that one only
        needs to modify the Args class to make changes to the training process. 

Usage:  This script should always reside in the root directory   
        of the repositor so that any path is the cwd directory 
        plus some arguement. Run this script from the project root!

"""

import torch
import os
from ultralytics import YOLO
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should show 11.8 or similar

class Args():
    def __init__(self):
        self.base_path = os.getcwd() 
        self.base_model = 'yolo11n.pt'

        # Training arguments (inputs to model.train)
        self.yaml_file_path = os.path.join(self.base_path, 'data/fish_dataset.yaml') # Insert name of dataset yaml here
        self.epochs = 300 
        self.images = 640
        self.batch = 8                  # How many images go into the gpu at once (adjust as needed)
        self.initial_lr = 0.001         # Initial learning rate (for Cosine Annealing)
        self.final_lr = 0.0001          # Final learning rate 
        self.optimizer = 'AdamW'        # Best performance 
        self.patients = 10              # Early stopping if no improvements after 10 epochs
        self.save_period = 10           # Save model checkpoint after X epochs
        self.augment = True             # Enable data augmentation 
        self.mosaic = True              # Use mosaic augmentation 
        self.mixup = True               # Use mixup augmentation 
        self.cos_lr = True              # Cosing annealing learning rate
        self.project_dir = 'runs/n_logs_v1'     # Where logs and weights are saved 

        # Saving arguments (how should things be named and where should they go)
        self.save_location = os.path.join(self.base_path, '/models/new')
        self.model_save_name = 'yolo11n_2016_fish_v3.pt'

        # Optional formats 
        self.save_onnx = False
        self.save_ncnn = False 
        self.save_tflite = False 

args = Args() # Instance of the argument holding class 


# Ensure the script uses the correct GPU (MODIFY FOR DUKE)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the smaller YOLO11 model
small_model = YOLO(args.base_model) # YOLO("yolo11X.pt") where X = n, m, or l 

# Move the model to the correct device
small_model.model.to(device)

# Freeze the first few layers for the first 10 epochs for better fine-tuning
# OPTIONAL (EXPERIMENT WITH THIS)
for param in small_model.model.model.parameters():
    param.requires_grad = False  # Freeze all layers initially

# Training hyperparameters
small_model.train(
    data=args.yaml_file_path,
    epochs=args.epochs,
    imgsz=args.images,
    batch=args.batch,  # Adjust batch size based on GPU capacity
    lr0=args.initial_lr,  # Initial learning rate
    lrf=args.final_lr,  # Final learning rate (used for Cosine Annealing)
    optimizer=args.optimizer,  # Use AdamW optimizer for better performance
    device=device,
    patience=args.patients,  # Early stopping if no improvement after 10 epochs
    save_period=args.save_period,  # Save model checkpoint every 10 epochs
    augment=args.augment,  # Enable data augmentation
    mosaic=args.mosaic,  # Use mosaic augmentation
    mixup=args.mixup,   # Use MixUp augmentation
    cos_lr=args.cos_lr,  # Cosine annealing learning rate
    project=args.project_dir,  # TensorBoard logging directory
)

print("Training complete!")

# Unfreeze all layers after the initial phase
for param in small_model.model.model.parameters():
    param.requires_grad = True

# Save the trained model
trained_model_path = os.path.join(args.save_location, args.model_save_name)
small_model.save(trained_model_path)
print(f"Trained model saved to {trained_model_path}")

# Save the model weights separately for further use
weights_path = os.path.join(args.save_location, "yolo11l_fish_2016_v2.pth")
torch.save(small_model.model.state_dict(), weights_path)
print(f"Weights saved to {weights_path}")

# Evaluate model performance
metrics = small_model.val(data=args.yaml_path, device=device)
print(metrics)

# Export the trained model to ONNX format
if args.save_onnx:
    try:
        small_model.export(format="onnx")
        print("ONNX model exported successfully!")
    except Exception as e:
        print(f"ONNX export failed: {e}")

# # Export to TensorFlow Lite
if args.save_tflite:
    try:
        small_model.export(format="tflite")
        print("TFLite model exported successfully!")
    except Exception as e:
        print(f"TFLite export failed: {e}")

# # Export to TensorFlow Edge TPU
# try:
#     small_model.export(format="edgetpu")
#     print("Edge TPU model exported successfully!")
# except Exception as e:
#     print(f"Edge TPU export failed: {e}")

# Export to NCNN format
if args.save_ncnn:    
    try:
        small_model.export(format="ncnn")  # Creates .param and .bin files
        print("NCNN files exported successfully!")
    except Exception as e:
        print(f"NCNN export failed: {e}")

if args.save_onnx or args.save_tflite or args.save_ncnn:
    print("Model exports completed where possible.")

print("Program completo! ")