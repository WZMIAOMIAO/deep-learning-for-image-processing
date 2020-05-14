from model import GoogLeNet
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

im_height = 224
im_width = 224

# load image
img = Image.open("../tulip.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# scaling pixel value and normalize
img = ((np.array(img) / 255.) - 0.5) / 0.5

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = GoogLeNet(class_num=5, aux_logits=False)
model.summary()
# model.load_weights("./save_weights/myGoogLenet.h5", by_name=True)  # h5 format
model.load_weights("./save_weights/myGoogLeNet.ckpt")  # ckpt format
result = model.predict(img)
predict_class = np.argmax(result)
print(class_indict[str(predict_class)])
plt.show()
