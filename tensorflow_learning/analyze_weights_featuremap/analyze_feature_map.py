from alexnet_model import AlexNet_v1, AlexNet_v2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input

im_height = 224
im_width = 224

# load image
img = Image.open("../tulip.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))

# scaling pixel value to (0-1)
img = np.array(img) / 255.

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))


model = AlexNet_v1(class_num=5)  # functional api
# model = AlexNet_v2(class_num=5)  # subclass api
# model.build((None, 224, 224, 3))
# If `by_name` is False weights are loaded based on the network's topology.
model.load_weights("./myAlex.h5")
# model.load_weights("./submodel.h5")
# for layer in model.layers:
#     print(layer.name)
model.summary()
layers_name = ["conv2d", "conv2d_1"]

# functional API
try:
    input_node = model.input
    output_node = [model.get_layer(name=layer_name).output for layer_name in layers_name]
    model1 = Model(inputs=input_node, outputs=output_node)
    outputs = model1.predict(img)
    for index, feature_map in enumerate(outputs):
        # [N, H, W, C] -> [H, W, C]
        im = np.squeeze(feature_map)

        # show top 12 feature maps
        plt.figure()
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            # [H, W, C]
            plt.imshow(im[:, :, i], cmap='gray')
        plt.suptitle(layers_name[index])
        plt.show()
except Exception as e:
    print(e)

# subclasses API
# outputs = model.receive_feature_map(img, layers_name)
# for index, feature_maps in enumerate(outputs):
#     # [N, H, W, C] -> [H, W, C]
#     im = np.squeeze(feature_maps)
#
#     # show top 12 feature maps
#     plt.figure()
#     for i in range(12):
#         ax = plt.subplot(3, 4, i + 1)
#         # [H, W, C]
#         plt.imshow(im[:, :, i], cmap='gray')
#     plt.suptitle(layers_name[index])
#     plt.show()
