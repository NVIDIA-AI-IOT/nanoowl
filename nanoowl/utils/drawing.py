from PIL import Image
from PIL import ImageDraw, ImageFont
import math

_FONTS = [
    "/usr/share/fonts/truetype/tlwg/Sawasdee.ttf",
    "/usr/share/fonts/truetype/tlwg/TlwgMono.ttf"
]

def get_default_font(size):
    try:
        return ImageFont.truetype(font=_FONTS[1], size=size)
    except OSError:
        return ImageFont.load_default()

    
def draw_detection(
        image, 
        detection, 
        draw_bbox=True,
        draw_text=True, 
        font_scale=0.1
    ):

    draw = ImageDraw.Draw(image, "RGBA")


    x0, y0, x1, y1 = detection["bbox"]

    bbox_height = (y1 - y0)
    bbox_width = (x1 - x0)
    bbox_size = math.sqrt(bbox_height * bbox_width)

    if draw_bbox:
        # draw.rectangle(((x0, y0), (x1, y1)), fill=(118, 185, 0, 160))
        draw.rectangle(((x0, y0), (x1, y1)), outline=(118, 185, 0, 255), width=16)

    if draw_text:

        font_size = int(font_scale * bbox_size)
        x_text = int(x0 + 0.4 * font_size)
        y_text = int(y0 + 0.1 * font_size)

        font_size = max(18, font_size)

        draw.text(
            (x_text, y_text), 
            f'{detection["text"]}', 
            fill=(255, 255, 255, 255),
            font=get_default_font(font_size),
            stroke_width=1,
            stroke_fill=(255, 255, 255, 255)
        )

    return image

def draw_detections_raw(
        image, 
        detections, 
        draw_bbox=True
    ):

    draw = ImageDraw.Draw(image, "RGBA")


    for bbox in detections[0]['boxes']:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        bbox_height = (y1 - y0)
        bbox_width = (x1 - x0)

        draw.rectangle(((x0, y0), (x1, y1)), outline=(118, 185, 0, 255), width=16)

    return image