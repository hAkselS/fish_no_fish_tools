"""
File:   train_fnf_model.py

Spec:   Use this script to train yolo 11 nano, medium or larger models. 
        All arguements are stored in the Args class such that one only
        needs to modify the Args class to make changes to the training process. 

Usage:  This script should always reside in the root directory   
        of the repositor so that any path is the cwd directory 
        plus some arguement. 

"""

import torch
import os
from ultralytics import YOLO
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should show 11.8 or similar

class Args():
    def __init__(self):
        self.base_path = ''
        self.base_model = ''

        # Training arguments (inputs to model.train)
        self.yaml_path = ''
        self.base_model = ''
        self.epochs = 300 
        self.images = 640
        self.initial_lr = 0.001         # Initial learning rate (for Cosine Annealing)
        self.final_lr = 0.0001          # Final learning rate 
        self.optimizer = 'AdamW'        # Best performance 
        self.patients = 10              # Early stopping if no improvements after 10 epochs
        self.save_period = 10           # Save model checkpoint after X epochs
        self.augment = True             # Enable data augmentation 
        self.mosaic = True              # Use mosaic augmentation 
        self.mixup = True               # Use mixup augmentation 
        self.cos_lr = True              # Cosing annealing learning rate
        self.project_dir = 'l_logs'     # Where logs and weights are saved 

        # Saving arguments (how should things be named and where should they go)




# Ensure the script uses the correct GPU (MODIFY FOR DUKE)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set paths (WINDOWS PATH)
base_path = 'fish_no_fish/test_scripts/data_dump/'
yaml_file_path = os.path.join(base_path, 'fish_dataset.yaml')

# Load the smaller YOLO11 model
small_model = YOLO("yolo11l.pt") #YOLO("yolo11n.pt")

# Move the model to the correct device
small_model.model.to(device)

# Freeze the first few layers for the first 10 epochs for better fine-tuning
# OPTIONAL (EXPERIMENT WITH THIS)
for param in small_model.model.model.parameters():
    param.requires_grad = False  # Freeze all layers initially

# Training hyperparameters
small_model.train(
    data=yaml_file_path,
    epochs=300,
    imgsz=640,
    batch=8,  # Adjust batch size based on GPU capacity
    lr0=0.001,  # Initial learning rate
    lrf=0.0001,  # Final learning rate (used for Cosine Annealing)
    optimizer='AdamW',  # Use AdamW optimizer for better performance
    device=device,
    patience=20,  # Early stopping if no improvement after 10 epochs
    save_period=10,  # Save model checkpoint every 10 epochs
    augment=True,  # Enable data augmentation
    mosaic=True,  # Use mosaic augmentation
    mixup=True,   # Use MixUp augmentation
    cos_lr=True,  # Cosine annealing learning rate
    project='l_logs',  # TensorBoard logging directory
)

#     patience=10,  # Early stopping if no improvement after 10 epochs
print("Training complete!")

# Unfreeze all layers after the initial phase
for param in small_model.model.model.parameters():
    param.requires_grad = True

# Save the trained model
# trained_model_path = os.path.join(base_path, "yolo11n_fish_2016_v1.pt")
trained_model_path = os.path.join(base_path, "yolo11l_fish_2016_v2.pt")
small_model.save(trained_model_path)
print(f"Trained model saved to {trained_model_path}")

# Save the model weights separately for further use
weights_path = os.path.join(base_path, "yolo11l_fish_2016_v2.pth")
torch.save(small_model.model.state_dict(), weights_path)
print(f"Weights saved to {weights_path}")

# Evaluate model performance
metrics = small_model.val(data=yaml_file_path, device=device)
print(metrics)

# Export the trained model to ONNX format
try:
    small_model.export(format="onnx")
    print("ONNX model exported successfully!")
except Exception as e:
    print(f"ONNX export failed: {e}")

# # Export to TensorFlow Lite
# try:
#     small_model.export(format="tflite")
#     print("TFLite model exported successfully!")
# except Exception as e:
#     print(f"TFLite export failed: {e}")

# # Export to TensorFlow Edge TPU
# try:
#     small_model.export(format="edgetpu")
#     print("Edge TPU model exported successfully!")
# except Exception as e:
#     print(f"Edge TPU export failed: {e}")

# Export to NCNN format
try:
    small_model.export(format="ncnn")  # Creates .param and .bin files
    print("NCNN files exported successfully!")
except Exception as e:
    print(f"NCNN export failed: {e}")

print("Model exports completed where possible.")