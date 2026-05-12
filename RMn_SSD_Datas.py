import cv2
import numpy as np
import math

# -------------------------
# Load MobileNet-SSD (Caffe)
# -------------------------
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# class labels in MobileNet-SSD
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
VEHICLE_CLASSES = {"car", "bus", "motorbike", "bicycle"}  # classes treated as vehicles

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# -------------------------
# Parameters
# -------------------------
PIXEL_TO_METER = 0.05   # calibration factor
FRAME_FPS = 25.0
FRAME_AREA_M2 = None    # will set from image.shape if None

# Example lane regions (user must tune these to the road)
LANES = [
    [(0, 200), (640, 200), (640, 400), (0, 400)],  # lane 0
    [(0, 400), (640, 400), (640, 480), (0, 480)],  # lane 1
]

# storage for centroid history (to estimate speed)
last_positions = {}  # id -> (x,y)


def point_in_poly(point, poly):
    """Check if a point is inside polygon (ray casting)."""
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi)
        if intersect:
            inside = not inside
    return inside


def RMn_SSD_Feat(image):
    """
    Input: Raw image
    Output: vehicle_count, density, speeds (list), lane_estimate (dict)
    """
    global FRAME_AREA_M2, last_positions

    H, W = image.shape[:2]
    if FRAME_AREA_M2 is None:
        FRAME_AREA_M2 = (W * H) * (PIXEL_TO_METER ** 2)

    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    vehicle_count = 0
    speeds = []
    lane_estimate = {}
    rects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0:
            continue
        idx = int(detections[0, 0, i, 1])
        if idx >= len(CLASS_NAMES):
            continue
        label = CLASS_NAMES[idx]
        if label not in VEHICLE_CLASSES:
            continue

        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        centroid = (cX, cY)

        # assign lane
        lane_id = -1
        for li, lane_poly in enumerate(LANES):
            if point_in_poly(centroid, lane_poly):
                lane_id = li
                break

        # estimate speed from previous frame (if centroid exists)
        speed_m_s = 0.0
        if i in last_positions:
            (px, py) = last_positions[i]
            dist_m = math.hypot(cX - px, cY - py) * PIXEL_TO_METER
            speed_m_s = dist_m * FRAME_FPS
        last_positions[i] = centroid

        speeds.append(speed_m_s)
        lane_estimate[i] = lane_id
        rects.append((startX, startY, endX, endY))
        vehicle_count += 1

    # density = vehicles per square meter
    density = vehicle_count / FRAME_AREA_M2 if FRAME_AREA_M2 > 0 else 0

    return vehicle_count, density, speeds, lane_estimate
