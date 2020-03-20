from alexnet_model import AlexNet_v1, AlexNet_v2
import numpy as np
import matplotlib.pyplot as plt

model = AlexNet_v1(class_num=5)  # functional api
# model = AlexNet_v2(class_num=5)  # subclass api
# model.build((None, 224, 224, 3))
model.load_weights("./myAlex.h5")
# model.load_weights("./submodel.h5")
model.summary()
for layer in model.layers:
    for index, weight in enumerate(layer.weights):
        # [kernel_height, kernel_width, kernel_channel, kernel_number]
        weight_t = weight.numpy()
        # read a kernel information
        # k = weight_t[:, :, :, 0]

        # calculate mean, std, min, max
        weight_mean = weight_t.mean()
        weight_std = weight_t.std(ddof=1)
        weight_min = weight_t.min()
        weight_max = weight_t.max()
        print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                                   weight_std,
                                                                   weight_max,
                                                                   weight_min))

        # plot hist image
        plt.close()
        weight_vec = np.reshape(weight_t, [-1])
        plt.hist(weight_vec, bins=50)
        plt.title(weight.name)
        plt.show()