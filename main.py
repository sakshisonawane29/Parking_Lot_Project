import cv2
import pickle
import cvzone
import numpy as np
import time
import os
from datetime import datetime

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

# Load parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

# Get the video properties to create the output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
output_video = cv2.VideoWriter('parking_output.mp4', fourcc, fps, (frame_width, frame_height))

# Create a list of vehicle names for assignment (shorter names)
vehicle_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", 
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "T1", "T2", "V1", "V2", "S1", "S2"
]

# Dictionary to map each parking space to a specific vehicle name
# This simulates a system where specific spaces are assigned to specific vehicles
assigned_vehicles = {}

# Assign vehicles to parking spaces in order
for i, pos in enumerate(posList):
    vehicle_index = i % len(vehicle_names)  # Cycle through names if more spaces than names
    assigned_vehicles[pos] = vehicle_names[vehicle_index]

# For simulation, we'll assign fixed vehicle IDs to spaces rather than generating them dynamically
# This creates a more stable demonstration
fixed_vehicle_assignments = {}
for i, pos in enumerate(posList):
    # For simplicity, just use numbers as vehicle IDs
    fixed_vehicle_assignments[pos] = i + 1

# Dictionary to track the current vehicle in each space
current_vehicles = {pos: None for pos in posList}

# For alarm system
alarm_active = False
last_alarm_time = 0
ALARM_COOLDOWN = 3  # seconds
violation_count = 0

def trigger_alarm(pos, assigned, current):
    """Trigger an alarm when a different vehicle is detected."""
    global alarm_active, last_alarm_time
    current_time = time.time()
    
    # Only trigger alarm if cooldown period has elapsed
    if not alarm_active or (current_time - last_alarm_time > ALARM_COOLDOWN):
        print(f"⚠️ ALARM: Space for Vehicle {assigned} occupied by Vehicle {current}!")
        last_alarm_time = current_time
        alarm_active = True

def checkParkingSpace(imgPro, img_original):
    global violation_count
    
    spaceCounter = 0
    wrong_vehicle_counter = 0

    for pos in posList:
        x, y = pos

        # Get region of interest for processing
        imgCrop = imgPro[y:y + height, x:x + width]
        # Count non-zero pixels for occupancy detection
        count = cv2.countNonZero(imgCrop)

        # Get the assigned vehicle for this parking space
        assigned_vehicle = assigned_vehicles[pos]

        if count < 900:  # Space is empty
            color = (0, 255, 0)  # Green
            thickness = 2
            spaceCounter += 1
            # Reset current vehicle when space becomes empty
            current_vehicles[pos] = None
            
            # For empty spaces, just show reserved text (smaller and centered)
            text_pos = (x + width//2 - 15, y + height//2)
            cv2.putText(img, f"{assigned_vehicle}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        else:  # Space is occupied
            # For demo purposes, each space has a fixed "actual vehicle" 
            # that occupies it in this simulation
            actual_vehicle = fixed_vehicle_assignments[pos]
            
            # Only the first 15 spaces have the "correct" vehicle (for demo)
            # The rest have a different vehicle (simulating violations)
            is_correct = (list(posList).index(pos) < 15)
            
            if is_correct:  # Correct vehicle
                color = (0, 0, 255)  # Red for occupied with correct vehicle
                thickness = 2
                current_vehicles[pos] = assigned_vehicle
                # Don't display text for correct vehicles to reduce clutter
            else:  # Wrong vehicle
                color = (0, 140, 255)  # Orange for violation
                thickness = 2
                current_vehicles[pos] = f"{actual_vehicle}"
                wrong_vehicle_counter += 1
                
                # Just show a small number for wrong vehicle
                text_pos = (x + width//2 - 5, y + height//2)
                cv2.putText(img, f"{actual_vehicle}", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Only trigger actual alarm for some spaces to reduce noise
                if list(posList).index(pos) % 5 == 0:
                    trigger_alarm(pos, assigned_vehicle, actual_vehicle)

        # Draw rectangle for each parking space
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

    # Display counters with better positioning and size
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (50, 50), scale=2,
                      thickness=2, offset=10, colorR=(0,200,0))
    
    if wrong_vehicle_counter > 0:
        violation_count = max(violation_count, wrong_vehicle_counter)
        cvzone.putTextRect(img, f'Violations: {wrong_vehicle_counter}', (50, 100), scale=1.5,
                          thickness=2, offset=5, colorR=(0,140,255))

print("Processing video... Please wait.")
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break  # Stop when video ends instead of looping
    
    success, img = cap.read()
    if not success:
        break
    
    # Image processing pipeline
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Pass both processed and original images
    checkParkingSpace(imgDilate, img)
    
    # Write frame to output video
    output_video.write(img)
    
    # Display progress
    frame_count += 1
    if frame_count % 30 == 0:  # Update progress every 30 frames
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.1f}%")
    
    # Optional: Display the result (can be commented out for faster processing)
    cv2.imshow("Parking Space Detection", img)
    
    # Check for keypress
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources
print("Processing complete! Output saved to 'parking_output.mp4'")
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"\nParking Monitoring Summary:")
print(f"Total Parking Spaces: {len(posList)}")
print(f"Maximum Violations Detected: {violation_count}")
print(f"Output video saved as: parking_output.mp4")