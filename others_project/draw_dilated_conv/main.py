import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def dilated_conv_one_pixel(center: (int, int),
                           feature_map: np.ndarray,
                           k: int = 3,
                           r: int = 1,
                           v: int = 1):
    """
    膨胀卷积核中心在指定坐标center处时，统计哪些像素被利用到，
    并在利用到的像素位置处加上增量v
    Args:
        center: 膨胀卷积核中心的坐标
        feature_map: 记录每个像素使用次数的特征图
        k: 膨胀卷积核的kernel大小
        r: 膨胀卷积的dilation rate
        v: 使用次数增量
    """
    assert divmod(3, 2)[1] == 1

    # left-top: (x, y)
    left_top = (center[0] - ((k - 1) // 2) * r, center[1] - ((k - 1) // 2) * r)
    for i in range(k):
        for j in range(k):
            feature_map[left_top[1] + i * r][left_top[0] + j * r] += v


def dilated_conv_all_map(dilated_map: np.ndarray,
                         k: int = 3,
                         r: int = 1):
    """
    根据输出特征矩阵中哪些像素被使用以及使用次数，
    配合膨胀卷积k和r计算输入特征矩阵哪些像素被使用以及使用次数
    Args:
        dilated_map: 记录输出特征矩阵中每个像素被使用次数的特征图
        k: 膨胀卷积核的kernel大小
        r: 膨胀卷积的dilation rate
    """
    new_map = np.zeros_like(dilated_map)
    for i in range(dilated_map.shape[0]):
        for j in range(dilated_map.shape[1]):
            if dilated_map[i][j] > 0:
                dilated_conv_one_pixel((j, i), new_map, k=k, r=r, v=dilated_map[i][j])

    return new_map


def plot_map(matrix: np.ndarray):
    plt.figure()

    c_list = ['white', 'blue', 'red']
    new_cmp = LinearSegmentedColormap.from_list('chaos', c_list)
    plt.imshow(matrix, cmap=new_cmp)

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)

    # 显示color bar
    plt.colorbar()

    # 在图中标注数量
    thresh = 5
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(matrix[y, x])
            ax.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black")
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    plt.show()
    plt.close()


def main():
    # bottom to top
    dilated_rates = [1, 2, 3]
    # init feature map
    size = 31
    m = np.zeros(shape=(size, size), dtype=np.int32)
    center = size // 2
    m[center][center] = 1
    # print(m)
    # plot_map(m)

    for index, dilated_r in enumerate(dilated_rates[::-1]):
        new_map = dilated_conv_all_map(m, r=dilated_r)
        m = new_map
    print(m)
    plot_map(m)


if __name__ == '__main__':
    main()
