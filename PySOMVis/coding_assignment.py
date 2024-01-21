import numpy as np
import os
from SOMToolBox_Parse import SOMToolBox_Parse


def TopoProd(_m, _n, _weights, k=None):  # noqa
    if k is None:
        k = len(_weights) - 1
    k = min(k, len(_weights) - 1)
    output_space = np.array(np.meshgrid(np.arange(_m), np.arange(_n))).T.reshape(-1, 2) #  generates a 2D array where each row contains the X and Y coordinates of all points in a grid of shape _m by _n.
    P = np.zeros((_m, _n, 3))
    for i, inp_pos in enumerate(_weights):
        out_pos = output_space[i]
        knn_out = np.linalg.norm(output_space - out_pos, axis=1).argsort()
        knn_inp = np.linalg.norm(_weights - inp_pos, axis=1).argsort()

        # distortion in input space
        dist_inp_knn_out_pre = np.linalg.norm(_weights[knn_out] - inp_pos, axis=1)
        dist_inp_knn_out= dist_inp_knn_out_pre[1:]
        dist_inp_knn_inp = np.linalg.norm(_weights[knn_inp] - inp_pos, axis=1)[1:]
        q1 = dist_inp_knn_out[:k] / dist_inp_knn_inp[:k]

        # distortion in output space
        dist_out_knn_out = np.linalg.norm(output_space[knn_out] - out_pos, axis=1)[1:]
        dist_out_knn_inp = np.linalg.norm(output_space[knn_inp] - out_pos, axis=1)[1:]
        q2 = dist_out_knn_out[:k] / dist_out_knn_inp[:k]

        p1 = np.prod(q1) ** (1 / k)
        p2 = np.prod(q2) ** (1 / k)
        p3 = np.prod(q1 * q2) ** (1 / (2 * k))

        P[tuple(out_pos)] = p1, p2, p3

    return P


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    weights_path = os.path.join("datasets", "chainlink", "chainlink.wgt.gz")
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, weights_path)
    weights = SOMToolBox_Parse(filename).read_weight_file()
    tp = TopoProd(weights['xdim'], weights['ydim'], weights['arr'], k=1)
    N = len(weights['arr'])
    P = tp[..., 2].sum() / (N * (N - 1))
    print(P)
    # plot all components
    p1 = tp[..., 0]
    p2 = tp[..., 1]
    p3 = tp[..., 2]

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    sns.heatmap(p1, ax=axs[0])
    axs[0].set_title('tp[...,0]')
    sns.heatmap(p2, ax=axs[1])
    axs[1].set_title('tp[...,1]')
    sns.heatmap(p3, ax=axs[2])
    axs[2].set_title('tp[...,2]')
    plt.show()
