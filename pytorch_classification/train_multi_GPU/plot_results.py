import math
import matplotlib.pyplot as plt

x = [0, 1, 2, 3]
y = [9, 5.5, 3, 2]

plt.bar(x, y, align='center')
plt.xticks(range(len(x)), ['One-GPU', '2 GPUs', '4 GPUs', '8 GPUs'])
plt.ylim((0, 10))
for i, v in enumerate(y):
    plt.text(x=i, y=v + 0.1, s=str(v) + ' s', ha='center')
plt.xlabel('Using number of GPU device')
plt.ylabel('Training time per epoch (second)')
plt.show()
plt.close()

x = list(range(30))
no_SyncBatchNorm = [0.348, 0.495, 0.587, 0.554, 0.637,
                    0.622, 0.689, 0.673, 0.702, 0.717,
                    0.717, 0.69, 0.716, 0.696, 0.738,
                    0.75, 0.75, 0.66, 0.713, 0.758,
                    0.777, 0.777, 0.769, 0.792, 0.802,
                    0.807, 0.807, 0.804, 0.812, 0.811]

SyncBatchNorm = [0.283, 0.514, 0.531, 0.654, 0.671,
                 0.591, 0.621, 0.685, 0.701, 0.732,
                 0.701, 0.74, 0.667, 0.723, 0.745,
                 0.679, 0.738, 0.772, 0.764, 0.765,
                 0.764, 0.791, 0.818, 0.791, 0.807,
                 0.806, 0.811, 0.821, 0.833, 0.81]

plt.plot(x, no_SyncBatchNorm, label="No SyncBatchNorm")
plt.plot(x, SyncBatchNorm, label="SyncBatchNorm")
plt.xlabel('Training epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()


x = list(range(30))
single_gpu = [0.569, 0.576, 0.654, 0.648, 0.609,
              0.637, 0.699, 0.709, 0.715, 0.715,
              0.717, 0.724, 0.722, 0.731, 0.721,
              0.774, 0.751, 0.787, 0.78, 0.77,
              0.763, 0.803, 0.754, 0.796, 0.799,
              0.815, 0.793, 0.808, 0.811, 0.806]
plt.plot(x, single_gpu, color="black", label="Single GPU")
plt.plot(x, no_SyncBatchNorm, label="No SyncBatchNorm")
plt.plot(x, SyncBatchNorm, label="SyncBatchNorm")
plt.xlabel('Training epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()


# epochs = 30
# lrf = 0.1
# lf0 = lambda x: math.cos(x * math.pi / epochs)
# lf1 = lambda x: 1 + math.cos(x * math.pi / epochs)
# lf2 = lambda x: (1 + math.cos(x * math.pi / epochs)) / 2
# lf3 = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
# x = range(epochs)
# y0 = [lf0(epoch) for epoch in x]
# y1 = [lf1(epoch) for epoch in x]
# y2 = [lf2(epoch) for epoch in x]
# y3 = [lf3(epoch) for epoch in x]
# plt.subplot(2, 2, 1)
# plt.plot(x, y0)
# plt.hlines(1, 0, epochs-1, colors="r", linestyles="dashed")
# plt.hlines(-1, 0, epochs-1, colors="r", linestyles="dashed")
# plt.xlim((0, epochs-1))
#
# plt.subplot(2, 2, 2)
# plt.plot(x, y1)
# plt.hlines(2, 0, epochs-1, colors="r", linestyles="dashed")
# plt.hlines(0, 0, epochs-1, colors="r", linestyles="dashed")
# plt.xlim((0, epochs-1))
#
# plt.subplot(2, 2, 3)
# plt.plot(x, y2)
# plt.hlines(1, 0, epochs-1, colors="r", linestyles="dashed")
# plt.hlines(0, 0, epochs-1, colors="r", linestyles="dashed")
# plt.xlim((0, epochs-1))
#
# plt.subplot(2, 2, 4)
# plt.plot(x, y3)
# plt.hlines(1, 0, epochs-1, colors="r", linestyles="dashed")
# plt.hlines(lrf, 0, epochs-1, colors="r", linestyles="dashed")
# plt.text(epochs-1, y3[-1], "{}".format(round(y3[-1], 1)))
# plt.xlim((0, epochs-1))
#
# plt.show()
# plt.close()
