import pandas as pd
import matplotlib.pyplot as plt

'''Input name and fully qualified path to CSV below!'''
csv_file_name_and_path = '/home/gpu_enjoyer/fish_no_fish_tools/models/new/fnf_40_epoch_720sz(2!!)/results.csv'
'''^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'''

# Read the CSV file
try:
    df = pd.read_csv(csv_file_name_and_path)
except:
    print("Bad file name or path")


# Plot the metrics against epoch number
metrics = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
    'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
]

# Rainbow colors used for plotting
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

# plt.figure(figsize=(15, 10))
# for i, metric in enumerate(metrics):
#     plt.figure(figsize=(15, 10))
#     plt.plot(df['epoch'], df[metric], label=metric, color=colors[i % len(colors)])
#     plt.xlabel('Epoch')
#     plt.ylabel(metric)
#     title_string = metric + ' vs. Epoch'
#     plt.title(title_string)
#     plt.legend()
#     plt.grid(True)


# plt.show()

plt.figure(figsize=(15,10))
# plt.plot(df['Epoch'], df[])
    
# Calculate fitness
"""Referenced weights from: https://github.com/ultralytics/ultralytics/issues/14137"""

df["fitness"] = df["metrics/mAP50(B)"] * 0.1 + df["metrics/mAP50-95(B)"] * 0.9

# Find the epoch with the highest fitness
best_epoch = df['fitness'].idxmax() + 1

# Disclaimer
print("\nDISCLAIMER: the follow statistics indicate the models metrics during the 'best' epoch at which time YOLO saves the best.pt model weights.\n")

print(f"File used:{csv_file_name_and_path}\n")

# Print model stats at best epoch
print(f"Best model was saved at epoch: {best_epoch}")
print(f"mAP50-95 at best epoch = {df['metrics/mAP50-95(B)'][best_epoch]}")
print(f"mAP50 at best epoch = {df['metrics/mAP50(B)'][best_epoch]}")
print(f"Precision at best epoch: {df['metrics/precision(B)'][best_epoch]}")
print(f"Recall at best epoch: {df['metrics/recall(B)'][best_epoch]}")

# True for existing (01/28/2025) FNF model 
# Best model was saved at epoch: 38
# Precision at best epoch: 0.89509
# Recall at best epoch: 0.85339
# mAP50 at best epoch = 0.92893
# mAP50-95 at best epoch = 0.60977