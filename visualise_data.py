import numpy as np
import matplotlib.pyplot as plt

# Load a sample .npy file
sample_file = 'processed_data/lbaq5s_lips.npy'
lips = np.load(sample_file)

# Visualize a few frames
for i in range(min(5, lips.shape[0])):  # Display up to 5 frames
    plt.imshow(lips[i])
    plt.axis('off')
    plt.show()
