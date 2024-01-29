import numpy as np
import os
from SOMToolBox_Parse import SOMToolBox_Parse


def TopoProd(_m, _n, _weights, k=None):  # noqa
    # First we need to reshape the weights to a 2D array, if they are in shape (m, n, d),
    # like for example the minisom output.
    if _weights.ndim == 3:
        _weights = _weights.reshape(-1, _weights.shape[-1])

    # If k is not given, we use the maximum possible value.
    if k is None:
        k = len(_weights) - 1

    # If k is larger than the number of weights, we use the maximum possible value.
    k = min(k, len(_weights) - 1)

    # generates a 2D array where each row contains the X and Y coordinates of all points in a grid of shape _m by _n.
    # this represents the output space (grid) of the SOM.
    output_space = np.array(np.meshgrid(np.arange(_m), np.arange(_n))).T.reshape(-1, 2)

    # initialize the 3D array that will contain the topographic product components for each unit in the SOM.
    P = np.zeros((_m, _n, 3))  # noqa

    # for each unit in the SOM, calculate the topographic product components.
    for i, inp_pos in enumerate(_weights):
        # get the position of the current unit in the output space.
        out_pos = output_space[i]

        # in the following we search the k nearest neighbors of the current unit in the input space and the output space
        # to do so we first calculate the Euclidean distance between the current unit and all other units
        # by subtracting the current position vector from all other position vectors element wise
        # and calculating the Euclidean norm of the resulting vectors using np.linalg.norm.
        # then we sort the resulting array of distances and get the indices of the resulting ordering.
        # We drop the first index, because it is the index of the current unit itself.
        # get the indices of all neighbors of the current unit, sorted by proximity in the output space.
        knn_out = np.linalg.norm(output_space - out_pos, axis=1).argsort()[1:]
        # get the indices of all neighbors of the current unit, sorted by proximity in the input space.
        knn_inp = np.linalg.norm(_weights - inp_pos, axis=1).argsort()[1:]

        # Next we calculate the distortion in the input space and the output space for the k nearest neighbors
        # of the current unit.
        # we use the calculated indices to get the input positions (weights) of the k nearest neighbors
        # then we calculate the Euclidean distance between the current unit and all k nearest neighbors
        # the resulting arrays hold the distances in input space between the current unit and all k nearest neighbors,
        # sorted by proximity in the input space and the output space respectively.
        distance_inp_knn_out = np.linalg.norm(_weights[knn_out] - inp_pos, axis=1)
        distance_inp_knn_inp = np.linalg.norm(_weights[knn_inp] - inp_pos, axis=1)
        # calculate the distortion in input space by dividing the distances in input space
        # by the distances in output space element wise. limit the number of neighbors to k, ignore the rest.
        q1 = distance_inp_knn_out[:k] / distance_inp_knn_inp[:k]
        # q1 now holds the distortion in input space, per neighbor.

        # we repeat this process for the output space
        dist_out_knn_out = np.linalg.norm(output_space[knn_out] - out_pos, axis=1)[1:]
        dist_out_knn_inp = np.linalg.norm(output_space[knn_inp] - out_pos, axis=1)[1:]
        q2 = dist_out_knn_out[:k] / dist_out_knn_inp[:k]
        # q2 now holds the distortion in output space, per neighbor.

        # to get the first component of the topographic product for the current unit,
        # we calculate the product of all distortion values per neighbor and take the k-th root.
        # for input space
        p1 = np.prod(q1) ** (1 / k)
        # for output space
        p2 = np.prod(q2) ** (1 / k)
        # for both spaces - the geometric mean of the distortion in input space and output space
        p3 = np.prod(q1 * q2) ** (1 / (2 * k))

        # finally we store the topographic product components for the current unit in the result array.
        P[tuple(out_pos)] = p1, p2, p3

    # After calculating the topographic product components for each unit in the SOM,
    # we return the resulting 3D array, which can then be plotted as a heatmap (3 heatmaps actually).
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
