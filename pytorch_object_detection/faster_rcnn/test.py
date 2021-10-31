import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

original_img = "/home/chaoc/Desktop/deep-learning-for-image-processing/data_set/test/IMAGES/1727.jpg"
with Image.open(original_img) as im:
    draw = ImageDraw.Draw(im)
    print((0, 0) + im.size)
    draw.line((0, 0) + im.size, fill=128)
    # draw.line((0, im.size[1], im.size[0], 0), fill=128)

    im.save("abc.jpg")
