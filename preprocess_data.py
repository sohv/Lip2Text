import cv2
import numpy as np
import os
from tqdm import tqdm
import dlib

def download_grid_corpus(base_path):
    """Create directory structure for GRID corpus data."""
    required_dirs = ['videos', 'align', 'processed_data']
    for dir in required_dirs:
        os.makedirs(os.path.join(base_path, dir), exist_ok=True)
    
    print("1. Video files (.mpg) - are in 'videos' directory")
    print("2. Alignment files (.align) - are in 'align' directory")

def extract_lip_region(frame, face_detector, landmark_predictor):
    """Extract lip region from the video frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray)
    if len(faces) == 0:
        return None
    
    # Get facial landmarks
    landmarks = landmark_predictor(gray, faces[0])

    # Extract lip landmarks (points 48-68 in dlib's facial landmarks)
    lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 68)])
    
    # Calculate bounding box for lips
    x_min = np.min(lip_points[:, 0]) - 10
    x_max = np.max(lip_points[:, 0]) + 10
    y_min = np.min(lip_points[:, 1]) - 10
    y_max = np.max(lip_points[:, 1]) + 10

    # Extract lip region
    lip_region = frame[y_min:y_max, x_min:x_max]
    
    # Resize to standard size
    lip_region = cv2.resize(lip_region, (96, 48))
    
    return lip_region

def load_alignment(align_path):
    """Load alignment file and return start and end times for each word."""
    alignments = []
    with open(align_path, 'r') as f:
        for line in f:
            start, end, word = line.strip().split()
            alignments.append((float(start), float(end), word))
    return alignments

def preprocess_video(video_path, output_path, face_detector, landmark_predictor, align_path):
    """Process a single video file with word alignment."""
    cap = cv2.VideoCapture(video_path)
    alignments = load_alignment(align_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

    frames = []
    frame_count = 0
    current_time = 0.0
    
    # Preprocess video with word alignment
    for start_time, end_time, word in alignments:
        while cap.isOpened() and current_time < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps  # Convert frame count to time in seconds
            if start_time <= current_time <= end_time:
                lip_region = extract_lip_region(frame, face_detector, landmark_predictor)
                if lip_region is not None:
                    frames.append(lip_region)

            frame_count += 1

    cap.release()

    # Save the processed frames
    np.save(output_path, np.array(frames))

def main():
    # Initialize face detection and landmark prediction
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    videos_dir = os.path.join(base_path, "videos")
    processed_dir = os.path.join(base_path, "processed_data")
    align_dir = os.path.join(base_path, "align")

    # Create directory structure and download data
    download_grid_corpus(base_path)

    # Process videos
    for video_file in tqdm(os.listdir(videos_dir)):
        if video_file.endswith('.mpg'):
            video_path = os.path.join(videos_dir, video_file)
            output_path = os.path.join(processed_dir, video_file.replace('.mpg', '_lips.npy'))
            align_path = os.path.join(align_dir, video_file.replace('.mpg', '.align'))
            
            try:
                if os.path.exists(align_path) and not os.path.exists(output_path):
                    preprocess_video(video_path, output_path, face_detector, landmark_predictor, align_path)
            except Exception as e:
                print(f"Error processing video: {video_file}")
                print(f"Alignment file: {align_path}")
                print(f"Error message: {e}\n")
                continue  # Skipping to the next video file

if __name__ == "__main__":
    main()