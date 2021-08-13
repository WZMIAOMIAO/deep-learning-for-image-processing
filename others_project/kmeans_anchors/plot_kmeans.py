import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

colors = np.array(['blue', 'black'])


def plot_clusters(data, cls, clusters, title=""):
    if cls is None:
        c = [colors[0]] * data.shape[0]
    else:
        c = colors[cls].tolist()

    plt.scatter(data[:, 0], data[:, 1], c=c)
    for i, clus in enumerate(clusters):
        plt.scatter(clus[0], clus[1], c='gold', marker='*', s=150)
    plt.title(title)
    plt.show()
    plt.close()


def distances(data, clusters):
    xy1 = data[:, None]  # [N,1,2]
    xy2 = clusters[None]  # [1,M,2]
    d = np.sum(np.power(xy2 - xy1, 2), axis=-1)
    return d


def k_means(data, k, dist=np.mean):
    """
    k-means methods
    Args:
        data: 需要聚类的data
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法
    """
    data_number = data.shape[0]
    last_nearest = np.zeros((data_number,))

    # init k clusters
    clusters = data[np.random.choice(data_number, k, replace=False)]
    print(f"random cluster: \n {clusters}")
    # plot
    plot_clusters(data, None, clusters, "random clusters")

    step = 0
    while True:
        d = distances(data, clusters)
        current_nearest = np.argmin(d, axis=1)

        # plot
        plot_clusters(data, current_nearest, clusters, f"step {step}")
        
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = dist(data[current_nearest == cluster], axis=0)
        last_nearest = current_nearest
        step += 1

    return clusters


def main():
    x1, y1 = [np.random.normal(loc=1., size=150) for _ in range(2)]
    x2, y2 = [np.random.normal(loc=5., size=150) for _ in range(2)]

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    plt.scatter(x, y, c='blue')
    plt.title("initial data")
    plt.show()
    plt.close()

    clusters = k_means(np.concatenate([x[:, None], y[:, None]], axis=-1), k=2)
    print(f"k-means fluster: \n {clusters}")


if __name__ == '__main__':
    main()
