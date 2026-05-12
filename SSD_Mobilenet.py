import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (small for speed, use yolov8m or yolov8l for accuracy)
model = YOLO("yolov8s.pt")

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}


def split_into_tiles(image, tile_size=640, overlap=0.2):
    """Split large image into overlapping tiles."""
    h, w, _ = image.shape
    stride = int(tile_size * (1 - overlap))
    tiles, positions = [], []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
                positions.append((x, y))
    return tiles, positions


def RMn_SSD_Images(image, tile_size=640):
    """Detect vehicles in aerial images using YOLOv8 + tiling."""
    annotated_image = image.copy()
    vehicle_count = 0

    # Split into tiles
    tiles, positions = split_into_tiles(image, tile_size=tile_size)

    for tile, (x_offset, y_offset) in zip(tiles, positions):
        results = model.predict(tile, imgsz=tile_size, conf=0.25, verbose=False)[0]

        for result in results.boxes:
            class_id = int(result.cls.item())
            class_name = model.names[class_id]

            if class_name in VEHICLE_CLASSES:
                vehicle_count += 1

                # Get coordinates and shift back to original image space
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{class_name}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

    return annotated_image, vehicle_count
