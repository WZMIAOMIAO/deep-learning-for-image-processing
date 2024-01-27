from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    # ... (rest of the colors)
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_text(draw, box, cls, score, category_index, color, font='arial.ttf', font_size=24):
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    text_width, text_height = draw.textsize(display_str, font=font)
    margin = np.ceil(0.05 * text_width)

    if top > text_height:
        text_location = (left, top - text_height)
    else:
        text_location = (left, bottom)

    draw.rectangle([text_location, (left + text_width + 2 * margin, text_location[1] + text_height)], fill=color)
    draw.text((left + margin, text_location[1]), display_str, fill='black', font=font)

def draw_masks(image, masks, colors, thresh=0.7, alpha=0.5):
    masks = np.where(masks > thresh, True, False)
    img_to_draw = np.copy(np.array(image))
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color
    out = np.array(image) * (1 - alpha) + img_to_draw * alpha
    return Image.fromarray(out.astype(np.uint8))

def draw_objs(image, boxes=None, classes=None, scores=None, masks=None, category_index=None, 
              box_thresh=0.1, mask_thresh=0.5, line_thickness=8, font='arial.ttf', 
              font_size=24, draw_boxes_on_image=True, draw_masks_on_image=True):
    idxs = np.greater(scores, box_thresh)
    boxes, classes, scores = boxes[idxs], classes[idxs], scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    if draw_boxes_on_image:
        draw = ImageDraw.Draw(image)
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=line_thickness)
            draw_text(draw, box, int(cls), float(score), category_index, color, font, font_size)

    if draw_masks_on_image and masks is not None:
        image = draw_masks(image, masks, colors, mask_thresh)

    return image
