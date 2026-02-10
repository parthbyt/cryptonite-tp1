from collections import deque
import cv2
import numpy as np
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Video Details
pts = deque(maxlen=10)
vs = cv2.VideoCapture("./CV/input.mp4")

width  = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = vs.get(cv2.CAP_PROP_FPS)
frame_time = 1/fps

# Person Detection - MobileNet SSD
person_model = YOLO("yolov8n.pt")

# Counting people in court only
court_roi = np.array([
    [300, 200],  # Top-Left point
    [1000, 200], # Top-Right point
    [1200, 700], # Bottom-Right point
    [100, 700]   # Bottom-Left point
], np.int32)
court_roi = court_roi.reshape((-1, 1, 2))

# Graph Plotting
all_trajectories = [] 
current_throw = []    
frames_missing = 0    
MAX_MISSING_FRAMES = 15 

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
out = cv2.VideoWriter('./CV/output.mp4', fourcc, fps, (width, height))

fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)


# Detect Persons Function
def detect_persons(frame):
    results = person_model(
        frame,
        conf=0.3,
        iou=0.5,
        device="cpu",
        verbose=False
    )

    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                feet_x = int((x1 + x2) / 2)
                feet_y = y2 
                feet_point = (feet_x, feet_y)

                result = cv2.pointPolygonTest(court_roi, feet_point, False)

                if result >0:
                    boxes.append((x1, y1, x2, y2))
            
    return boxes



while True:
    start = time.time()

    ret, frame = vs.read()

    if ret == False:
        break
    
    # Color Mask
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 40, 80])
    upper_yellow = np.array([40, 255, 255])

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7,7), np.uint8)

    yellow_dilated = cv2.dilate(yellow_mask, kernel)
    blue_dilated   = cv2.dilate(blue_mask, kernel)

    color_mask = cv2.bitwise_or(yellow_dilated, blue_dilated)

    # Motion Mask
    motion_mask = fgbg.apply(blurred)

    motion_mask = cv2.erode(motion_mask, None, iterations=2)
    motion_mask = cv2.dilate(motion_mask, None, iterations=1)
    
    mask = cv2.bitwise_and(color_mask, motion_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None

    for i in contours:
        # Ball Detection
        area =  cv2.contourArea(i)
        
        peri = cv2.arcLength(i, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.7:
            continue
        
        
        (x, y, w, h) = cv2.boundingRect(i)

        if y < height * 0.25:
            min_r, max_r = 4, 13
        elif y < height * 0.65:
            min_r, max_r = 8, 15
        else:
            min_r, max_r = 7, 17


        radius = max(w, h) / 2
        
        if not (min_r < radius < max_r):
            continue

        aspect_ratio = w / h

        if 0.9 < aspect_ratio < 1.2:
            # Ball Outline
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = ((int((w) / 2) + x) , int(((h) / 2))+ y)

            pts.appendleft(center)

    # Graphing 
    if center is not None:
        current_throw.append(center)
        frames_missing = 0 # Reset counter because we found the ball
    else:
        frames_missing += 1
        # If we lost the ball for >15 frames and we have data, save the previous throw
        if frames_missing > MAX_MISSING_FRAMES and len(current_throw) > 0:
            all_trajectories.append(current_throw)
            current_throw = []

    #Contrail Plotting
    for i in range(1, len(pts)):
        pt1 = pts[i - 1]
        pt2 = pts[i]

        if pt1 is None or pt2 is None:
            continue
        
        cv2.circle(frame, pt2, 4, (0, 0, 255), -1)

        d = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if d > 120:
            continue

        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)


    # Person Detection and Plotting
    persons = detect_persons(frame)

    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.putText(frame, f"Players: {len(persons)}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    #Writing to Video
    out.write(frame)
    cv2.imshow("window", frame)

    elapsed = time.time() - start
    delay = max(0, frame_time - elapsed)
    time.sleep(delay)

    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == 27:
        break

out.release()
vs.release()
cv2.destroyAllWindows()

# [Added] Graph Generation Logic
print("Generating trajectory graph...")
plt.figure(figsize=(10, 6))

# Plot each separate throw as a different line
for i, trajectory in enumerate(all_trajectories):
    # Only plot if we have enough points to make a line
    if len(trajectory) > 5:
        traj_array = np.array(trajectory)
        plt.plot(traj_array[:, 0], traj_array[:, 1], label=f'Throw {i+1}', linewidth=2)

plt.title("Volleyball Trajectory Analysis")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.xlim(0, width)
plt.ylim(height, 0) # Invert Y-axis to match video coordinates (0,0 is top-left)
plt.legend()
plt.grid(True)

# Save the graph
output_graph_path = './CV/trajectory_graph.png'
plt.savefig(output_graph_path)
print(f"Graph saved to {output_graph_path}")
plt.show()