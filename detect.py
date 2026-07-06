import os
import sys
import time
import cv2
import torch
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv

# Enforce secure and explicit hardware device allocation
device = ('cuda' if torch.cuda.is_available() 
          else 'mps' if torch.backends.mps.is_available() 
          else 'cpu')
print(f"CRITICAL RESOURCE ALLOCATION -> Active Hardware Device: {device.upper()}")

# Load hidden environment variable tokens safely
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if not WANDB_API_KEY:
    raise ValueError("CRITICAL SECURITY ERROR: 'WANDB_API_KEY' not located in your local .env file!")

# Authenticate the remote experiment tracker programmatically
wandb.login(key=WANDB_API_KEY)

def record_inference_run(weights_path, output_filename):
    """Runs live camera tracking and automatically records an 8-second sample video."""
    if os.path.exists(weights_path):
        print(f"Loading custom optimized model weights from: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"Custom weights absent at '{weights_path}'. Loading generic base pretrained model...")
        model = YOLO("yolo26n-pose.pt")
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL ERROR: Cannot interface with webcam hardware index.")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30, (frame_width, frame_height))
    
    print(f"VIDEO RECORDER STARTED -> Target Path: {output_filename}")
    print("Recording will automatically close after 8 seconds. Press 'q' to abort early...")
    
    start_time = time.time()
    while (time.time() - start_time) < 8.0:
        ret, frame = cap.read()
        if not ret: break
        
        # Map OpenCV's native BGR layout to the network's trained RGB profile
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Explicitly scale the input canvas size to match spatial boundaries
        results = model.predict(rgb_frame, imgsz=640, device=device, verbose=False, conf=0.25)
        
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        cv2.imshow("YOLOv26 Live Stream Video Recorder", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"RECORDING COMPLETE: File saved successfully to {output_filename}\n")

TARGET_CUSTOM_WEIGHTS = "D:/Program Files/DS26/DLHW06/runs/pose/train_phase2/weights/best.pt"

if not os.path.exists(TARGET_CUSTOM_WEIGHTS):
    print(f"CRITICAL ERROR: Custom fine-tuned weights missing at '{TARGET_CUSTOM_WEIGHTS}'")
    print("Ensure your training loop ran completely and successfully saved its outputs.")
else:
    print("Target weights verified. Launching successful tracking video output recording...")
    record_inference_run(weights_path=TARGET_CUSTOM_WEIGHTS, output_filename="D:/Program Files/DS26/DLHW06/runs/pose/predict/successful_attempt_phase2.mp4")