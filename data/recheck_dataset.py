import numpy as np
import matplotlib.pyplot as plt
import random

# Load the standardized data
standardized_content = np.load("dataset/traindata.npy", allow_pickle=True).item()

# Basic overview
print(f"Total samples: {len(standardized_content)}")
print("Keys in the dataset:", standardized_content.keys())

# Check shapes and size
shapes = [value.shape for value in standardized_content.values()]
print("Sample shapes:", shapes)

# Check for consistency
all_shapes_same = all(shape == (75, 48, 96, 3) for shape in shapes)
if all_shapes_same:
    print("All samples have the correct shape.")
else:
    print("Warning: Not all samples have the correct shape.")

# Data type and range
for key, value in standardized_content.items():
    print(f"Key: {key}, Data type: {value.dtype}, Value range: {value.min()} to {value.max()}")

# Visual inspection
def visualize_sample(sample):
    num_frames = sample.shape[0]
    plt.figure(figsize=(15, 3))
    for i in range(min(5, num_frames)):  # Display the first 5 frames
        plt.subplot(1, 5, i + 1)
        plt.imshow(sample[i])
        plt.axis('off')
        plt.title(f"Frame {i + 1}")
    plt.show()

# Visualize random samples
random_keys = random.sample(list(standardized_content.keys()), 3)  # Change the number as needed
for key in random_keys:
    print(f"Visualizing sample: {key}")
    visualize_sample(standardized_content[key])

# Statistical summary
pixel_values = np.concatenate([value.flatten() for value in standardized_content.values()])
print(f"Pixel value statistics: Mean = {np.mean(pixel_values)}, Median = {np.median(pixel_values)}, Std = {np.std(pixel_values)}")

# Check for duplicates
duplicate_keys = [key for key in standardized_content.keys() if list(standardized_content.keys()).count(key) > 1]
if duplicate_keys:
    print(f"Warning: Found duplicate keys: {duplicate_keys}")
else:
    print("No duplicates found.")