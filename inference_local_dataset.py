'''
File:   interence_local_dataset.py

Spec:   This script it used to inference a large, local dataset
        and output the training results into ______ forma.
        Results are stored into unsupervised_annotations folder. 
'''

from ultralytics import YOLO
import os 
import torch 

class Args():
    '''A class to store arguments for easy access.'''
    def __init__(self):
        self.model_path = '/home/gpu_enjoyer/fish_no_fish_tools/models/accepted/fnf_v1/train3(fnf_candidate_2)/weights/best(used in fnf on MM).pt'
        self.dataset = ''
        self.inference_save_location = ''


args = Args() # Instance of arguement holding class 

# Ensure the script uses the correct GPU (MODIFY FOR DUKE)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(args.model_path)

