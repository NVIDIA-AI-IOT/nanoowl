from PIL import Image
from PIL import ImageDraw
import math


def draw_detections(
        image, 
        detections
    ):

    draw = ImageDraw.Draw(image, "RGBA")


    for bbox in detections[0]['boxes']:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        bbox_height = (y1 - y0)
        bbox_width = (x1 - x0)

        draw.rectangle(((x0, y0), (x1, y1)), outline=(118, 185, 0, 255), width=4)

    return image