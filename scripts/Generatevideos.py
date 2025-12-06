import os
import cv2
import torch
import numpy as np
import torchvision
from tqdm import tqdm

# Load your trained model
print("Loading trained model...")
model_path = 'output/segmentation/deeplabv3_resnet50_random/best.pt'
checkpoint = torch.load(model_path, map_location='cpu')

# Initialize model
print("Initializing model architecture...")
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=0.5, num_classes=1)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Create output directory
output_dir = 'segmentation_videos'
os.makedirs(output_dir, exist_ok=True)

# Data directory
data_dir = 'C:/Users/msi/Documents/Endoscopy/data'
video_dir = os.path.join(data_dir, 'Videos')

# Get list of all videos (first 100 only)
all_videos = [f for f in os.listdir(video_dir) if f.endswith('.avi')][:100]

print(f"\nGenerating segmentation videos for {len(all_videos)} samples...")

for video_filename in tqdm(all_videos):
    try:
        # Load original video
        video_path = os.path.join(video_dir, video_filename)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f" Could not open {video_filename}")
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = os.path.join(output_dir, f'seg_{video_filename}')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prepare frame for model (resize to model input size)
            frame_resized = cv2.resize(frame, (112, 112))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            
            # Get segmentation prediction
            with torch.no_grad():
                pred = model(frame_tensor)['out']
                pred_mask = torch.sigmoid(pred[0, 0]).cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
                
                # Resize mask back to original size
                pred_mask_resized = cv2.resize(pred_mask, (width, height))
                
                # Create colored overlay (blue for left ventricle)
                overlay = frame.copy()
                overlay[pred_mask_resized == 1] = [255, 100, 0]  # Blue color (BGR)
                
                # Blend original and overlay
                result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Write frame
                out.write(result)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
    except Exception as e:
        print(f" Error processing {video_filename}: {str(e)}")
        continue

print(f"\n Done! Videos saved in '{output_dir}/' folder")
print(f"Location: {os.path.abspath(output_dir)}")
