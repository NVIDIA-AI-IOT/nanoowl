from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.font_manager

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
        text_padding=(32, 8),
        font_scale=0.15
    ):

    draw = ImageDraw.Draw(image, "RGBA")


    x0, y0, x1, y1 = detection["bbox"]

    bbox_height = (y1 - y0)

    if draw_bbox:
        draw.rectangle(((x0, y0), (x1, y1)), fill=(118, 185, 0, 50))
        draw.rectangle(((x0, y0), (x1, y1)), outline=(255, 255, 255, 255), width=2)

    if draw_text:

        font_size = int(font_scale * bbox_height)
        x_text = int(x0 + 0.4 * font_size)
        y_text = int(y0 + 0.1 * font_size)

        draw.text(
            (x_text, y_text), 
            detection["text"], 
            fill=(255, 255, 255, 255),
            font=get_default_font(font_size)
        )

    return image