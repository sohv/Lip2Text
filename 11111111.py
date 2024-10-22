import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def read_align_file(file_path):
    """Read alignment file and return text content"""
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading align file {file_path}: {str(e)}")
        return None

def extract_frames(video_path, desired_frames=75):
    """Extract frames from video and preprocess them"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
            
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to skip to get desired number of frames
        step = max(total_frames // desired_frames, 1)
        
        for frame_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract mouth region (you might need to adjust these coordinates)
                height, width = gray.shape
                mouth_region = gray[height//2:, width//3:2*width//3]
                
                # Resize to standard size
                mouth_region = cv2.resize(mouth_region, (28, 28))
                
                frames.append(mouth_region)
                
                if len(frames) >= desired_frames:
                    break
                    
        cap.release()
        
        # If we couldn't extract any frames, return None
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return None
            
        # Pad if we don't have enough frames
        while len(frames) < desired_frames:
            frames.append(np.zeros((28, 28), dtype=np.uint8))
            
        return np.array(frames)
        
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def process_dataset(align_path, video_path, output_path):
    """Process the entire dataset and create CSV files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    data = []
    errors = []
    
    # Get all video files first
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mpg')]
    
    print(f"Found {len(video_files)} video files")
    print("Processing dataset...")
    
    for video_file in tqdm(video_files):
        try:
            base_name = video_file.replace('.mpg', '')
            align_file = base_name + '.align'
            
            video_file_path = os.path.join(video_path, video_file)
            align_file_path = os.path.join(align_path, align_file)
            
            # Check if alignment file exists
            if not os.path.exists(align_file_path):
                print(f"Warning: Alignment file missing for {video_file}. Skipping...")
                errors.append(f"Missing alignment: {video_file}")
                continue
            
            # Read alignment file
            text = read_align_file(align_file_path)
            if text is None:
                errors.append(f"Alignment read error: {video_file}")
                continue
            
            # Extract frames from video
            frames = extract_frames(video_file_path)
            if frames is None:
                errors.append(f"Frame extraction error: {video_file}")
                continue
            
            # Save frames as numpy array
            frames_output_path = os.path.join(output_path, f"{base_name}.npy")
            np.save(frames_output_path, frames)
            
            # Add to dataset information
            data.append({
                'file_id': base_name,
                'text': text,
                'frames_path': frames_output_path,
                'n_frames': len(frames)
            })
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            errors.append(f"Processing error {video_file}: {str(e)}")
            continue
    
    # Create DataFrame and save as CSV
    if data:
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_path, 'dataset.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nDataset CSV saved to {csv_path}")
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"Total samples processed: {len(df)}")
        print(f"Average text length: {df['text'].str.len().mean():.2f} characters")
        print(f"Average frames per video: {df['n_frames'].mean():.2f}")
    else:
        print("No data was successfully processed!")
    
    # Save error log
    if errors:
        error_path = os.path.join(output_path, 'processing_errors.txt')
        with open(error_path, 'w') as f:
            f.write('\n'.join(errors))
        print(f"\nProcessing errors saved to {error_path}")

if __name__ == "__main__":
    # Your specified paths
    ALIGN_PATH = r"C:\Users\Ashish Mahendran\Documents\New folder (3)\lipreading_project\align"
    VIDEO_PATH = r"C:\Users\Ashish Mahendran\Documents\New folder (3)\lipreading_project\videos"
    OUTPUT_PATH = r"C:\Users\Ashish Mahendran\Documents\New folder (4)"
    
    # Process the dataset
    process_dataset(ALIGN_PATH, VIDEO_PATH, OUTPUT_PATH)