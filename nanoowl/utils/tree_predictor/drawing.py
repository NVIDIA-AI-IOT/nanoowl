import PIL.Image
import PIL.ImageDraw
import cv2
from .tree import Tree
from .predictor import TreeDetection
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def draw_tree_detections(image, detections: List[TreeDetection], tree: Tree, draw_text=True, num_colors=8):
    is_pil = not isinstance(image, np.ndarray)
    if is_pil:
        image = np.asarray(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    colors = get_colors(num_colors)
    label_map = tree.get_buffer_label_map()
    label_depths = tree.get_buffer_depth_map()
    for detection in detections.values():
        box = [int(x) for x in detection.box]
        pt0 = (box[0], box[1])
        pt1 = (box[2], box[3])
        box_depth = min(label_depths[i] for i in detection.labels)
        cv2.rectangle(
            image,
            pt0,
            pt1,
            colors[box_depth % num_colors],
            4
        )
        if draw_text:
            offset_y = 12
            offset_x = 0
            for label in detection.labels:
                label_text = label_map[label]
                if label_text is not None:
                    label_text = label_text[2]
                    cv2.putText(
                        image,
                        label_text,
                        (box[0] + offset_x, box[1] + offset_y),
                        font,
                        font_scale,
                        colors[label % num_colors],
                        2,# thickness
                        cv2.LINE_AA
                    )
                    offset_y += 18
    if is_pil:
        image = PIL.Image.fromarray(image)
    return image