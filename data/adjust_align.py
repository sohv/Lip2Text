import numpy as np

standardized_content = np.load("datasett/traindata.npy", allow_pickle=True).item()
alignments = np.load("datasett/train_labels.npy", allow_pickle=True).item()

# new dictionary to hold adjusted alignments
adjusted_alignments = {}

# target length for padding
padded_length = 75 

# Iterate through the standardized content
for key, value in standardized_content.items():
    original_length = value.shape[0]

    original_alignment = alignments[key]

    if original_length < padded_length:
        adjusted_alignment = np.pad(original_alignment, (0, padded_length - original_length), mode='constant', constant_values=-1)  # Use -1 or another value for padded frames
        adjusted_alignments[key] = adjusted_alignment
    elif original_length > padded_length:
        adjusted_alignment = original_alignment[:padded_length]
        adjusted_alignments[key] = adjusted_alignment
    else:
        adjusted_alignments[key] = original_alignment

np.save("datasett/alignments.npy", adjusted_alignments)