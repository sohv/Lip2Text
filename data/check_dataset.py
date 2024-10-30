import numpy as np
import matplotlib.pyplot as plt
import random

data = np.load("dataset/train_data.npy", allow_pickle=True)

print(f"Extracted content type: {type(data.item())}")

# Unpack the object stored in the scalar
content = data.item()
print(f"Content type: {type(content)}")
print(f"Total samples: {len(content)}")

# Inspect the first few entries
for i, (key, value) in enumerate(content.items()):
    print(f"Sample {i}: Key = {key}, Value shape = {value.shape}, Data type: {value.dtype}")
    print(f"First frame min/max pixel values: {value[0].min()}, {value[0].max()}")
    if i >= 5:
        break

consistent = True
expected_shape = None

for key, value in content.items():
    if expected_shape is None:
        expected_shape = value.shape
    if value.shape != expected_shape:
        print(f"Inconsistent shape for key {key}: {value.shape}")
        consistent = False

if consistent:
    print("All samples have consistent shapes.")
else:
    print("Warning: Inconsistent shapes found.")

# visualizing a sample
def visualize_sample(sample):
    num_frames = sample.shape[0]

    plt.figure(figsize=(15, 3))
    for i in range(min(5, num_frames)):  # displaying the first 5 frames
        plt.subplot(1, 5, i + 1)
        plt.imshow(sample[i])
        plt.axis('off')
        plt.title(f"Frame {i + 1}")

    plt.show()

# display a random sample from the dictionary
random_key = random.choice(list(content.keys()))
print(f"Visualizing sample: {random_key}")
visualize_sample(content[random_key])
