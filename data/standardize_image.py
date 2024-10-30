import numpy as np

# Load the .npy file
data = np.load("dataset/train_data.npy", allow_pickle=True)

print(f"Extracted content type: {type(data.item())}")

content = data.item()
print(f"Content type: {type(content)}")

# desired shape of train image
desired_shape = (75, 48, 96, 3)

standardized_content = {}

for key, value in content.items():
    if isinstance(value, np.ndarray):
        if value.shape[0] < desired_shape[0]:
            padded_sample = np.pad(value, 
                                   ((0, desired_shape[0] - value.shape[0]), (0, 0), (0, 0), (0, 0)), 
                                   mode='constant')
            standardized_content[key] = padded_sample
            print(f"Padded sample for key {key}: {padded_sample.shape}")
        elif value.shape[0] > desired_shape[0]:
            trimmed_sample = value[:desired_shape[0]]
            standardized_content[key] = trimmed_sample
            print(f"Trimmed sample for key {key}: {trimmed_sample.shape}")
        else:
            standardized_content[key] = value
            print(f"Sample for key {key} is already standardized: {value.shape}")
    else:
        print(f"Warning: Value for key {key} is not a NumPy array.")

print(f"Total standardized samples: {len(standardized_content)}")
save_path = "dataset/standardized_train_data.npy"
np.save(save_path, standardized_content, allow_pickle=True)
print(f"Standardized data saved to {save_path}")
