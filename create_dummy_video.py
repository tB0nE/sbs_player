import numpy as np
import cv2

def create_dummy_video(filename="dummy_test.mp4", duration_sec=5, fps=30, width=640, height=360):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame_idx in range(duration_sec * fps):
        # Create a frame with a moving circle to simulate depth/foreground elements
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw background pattern (horizontal lines)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (50, 50, 50), 2)
            
        # Draw a moving foreground object (circle)
        cx = int(100 + (frame_idx * 3) % (width - 200))
        cy = int(height / 2 + np.sin(frame_idx / 10.0) * 50)
        cv2.circle(frame, (cx, cy), 60, (0, 0, 255), -1) # Red circle (foreground)
        
        # Add a text label
        cv2.putText(frame, f"Frame {frame_idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"[Info] Dummy video '{filename}' created.")

if __name__ == "__main__":
    create_dummy_video()
