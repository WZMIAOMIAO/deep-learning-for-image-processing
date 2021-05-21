# vgg16(D)
model = [[3, 1],
         [3, 1],
         [2, 2],  # maxpool
         [3, 1],
         [3, 1],
         [2, 2],  # maxpool
         [3, 1],
         [3, 1],
         [3, 1],
         [2, 2],  # maxpool
         [3, 1],
         [3, 1],
         [3, 1],
         [2, 2],  # maxpool
         [3, 1],
         [3, 1],
         [3, 1]]

field = model[-1][0]
for kernel, stride in model[::-1]:
    field = (field - 1) * stride + kernel
print(field)  # 228
