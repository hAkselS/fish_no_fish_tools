import pandas as pd
import matplotlib.pyplot as plt

'''Innput name and fully qualified path to CSV below!'''
csv_file_name_and_path = '/home/gpu_enjoyer/fish_no_fish_tools/runs/n_logs_v1/train/results.csv'

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
for i, metric in enumerate(metrics):
    plt.figure(figsize=(15, 10))
    plt.plot(df['epoch'], df[metric], label=metric, color=colors[i % len(colors)])
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    title_string = metric + ' vs. Epoch'
    plt.title(title_string)
    plt.legend()
    plt.grid(True)


plt.show()

print(metrics)

plt.figure(figsize=(15,10))
# plt.plot(df['Epoch'], df[])

# Print out the max mAP scores and the epoch they were achieved at
max_map50 = df['metrics/mAP50(B)'].max()
max_map50_epoch = df[df['metrics/mAP50(B)'] == max_map50]['epoch'].values[0]

max_map50_95 = df['metrics/mAP50-95(B)'].max()
max_map50_95_epoch = df[df['metrics/mAP50-95(B)'] == max_map50_95]['epoch'].values[0]

print(f"Max mAP50: {max_map50} at epoch {max_map50_epoch}")
print(f"Max mAP50-95: {max_map50_95} at epoch {max_map50_95_epoch}")