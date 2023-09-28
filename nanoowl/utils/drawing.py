import cv2
from PIL import Image, ImageDraw
import math


def draw_detections(
        image, 
        detections,
        color=(185, 186, 0, 255),
        thickness=3
    ):

    draw = ImageDraw.Draw(image, "RGBA")


    for bbox in detections[0]['boxes']:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        bbox_height = (y1 - y0)
        bbox_width = (x1 - x0)

        draw.rectangle(((x0, y0), (x1, y1)), outline=color, width=thickness)

    return image


def draw_detections_cv2(
        image,
        detections,
        color=(0, 186, 118),
        thickness=3
    ):


    for bbox in detections[0]['boxes']:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        bbox_height = (y1 - y0)
        bbox_width = (x1 - x0)

        image = cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

    return image