import cv2
from ultralytics import YOLO
import numpy as np
import os

def process_source(source_path, model):
    # Check if source is image or video
    ext = os.path.splitext(source_path)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

    if ext in image_extensions:
        # Process Image
        print(f"Processing image: {source_path}")
        frame = cv2.imread(source_path)
        if frame is None:
            print(f"Error: Could not read image {source_path}")
            return

        # Use predict for single images (no tracking needed)
        results = model.predict(frame, conf=0.6, iou=0.4, agnostic_nms=True, verbose=False)
        
        count = 0
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            count = len(boxes)
            for box in boxes:
                # Draw bounding box (Purple color BGR: 128, 0, 128)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (128, 0, 128), 2)

        # Display Total Count
        cv2.putText(
            frame,
            f"Total Rods: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        
        cv2.imshow("Steel Rod Detection", frame)
        print(f"Finished. Total rods detected: {count}")
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif ext in video_extensions:
        # Process Video with Tracking
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source_path}")
            return

        count_of_rods = set()
        print(f"Processing video: {source_path}... Press 'q' to stop.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run tracking on the current frame
            results = model.track(frame, persist=True, verbose=False, conf=0.6, iou=0.4, agnostic_nms=True)

            if results[0].boxes.id is not None:
                # Extract IDs and boxes
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, id in zip(boxes, ids):
                    # Update the unique rod count set
                    count_of_rods.add(id)

                    # Draw bounding box (Purple color BGR: 128, 0, 128)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (128, 0, 128), 2)

            # Display Total Count on frame
            total_unique_count = len(count_of_rods)
            cv2.putText(
                frame,
                f"Total Unique Rods: {total_unique_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

            # Show the frame
            cv2.imshow("Steel Rod Detection and Tracking", frame)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished. Total unique rods detected: {len(count_of_rods)}")
    else:
        print(f"Error: Unsupported file format '{ext}'. Please use an image or video file.")

def main():
    # Load the trained model
    model = YOLO("best.pt")

    # You can change this path to any image or video file
    source_path = "istockphoto-1302633333-640_adpp_is.mp4" 
    
    if os.path.exists(source_path):
        process_source(source_path, model)
    else:
        print(f"Error: File '{source_path}' not found.")

if __name__ == "__main__":
    main()
