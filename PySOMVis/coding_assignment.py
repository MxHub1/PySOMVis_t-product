import numpy as np
import os

from minisom import MiniSom
from SOMToolBox_Parse import SOMToolBox_Parse


def TopoProd(_m, _n, _weights, k=None):  # noqa
    """
    Calculates the topographic product components for each unit in a Self-Organizing Map (SOM).

    This function calculates the topographic product components for each unit in a SOM. The topographic product is a
    measure of the topographic quality of a SOM. It is calculated based on the distortions in the input and output
    spaces for the k nearest neighbors of each unit in the SOM.

    Args:
        _m (int): The number of rows in the SOM grid.
        _n (int): The number of columns in the SOM grid.
        _weights (numpy.ndarray): The weights of the SOM units. If the weights are in shape (m, n, d), they are reshaped
                                   to a 2D array. The weights represent the position of the units in the input space.
        k (int, optional): The number of nearest neighbors to consider when calculating the topographic product.
                            If not provided, the maximum possible value is used.

    Returns:
        numpy.ndarray: A 3D array of shape (_m, _n, 3) containing the topographic product components for each unit in
                       the SOM. The first component is the distortion in the input space, the second component is the
                       distortion in the output space, and the third component is the geometric mean of the distortion
                       in the input space and the output space.
    """
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
        distance_out_knn_out = np.linalg.norm(output_space[knn_out] - out_pos, axis=1)
        distance_out_knn_inp = np.linalg.norm(output_space[knn_inp] - out_pos, axis=1)
        q2 = distance_out_knn_out[:k] / distance_out_knn_inp[:k]
        # q2 now holds the distortion in output space, per neighbor.

        # to get the first component of the topographic product for the current unit,
        # we calculate the product of all distortion values per neighbor and take the k-th root.
        # for input space
        prod_1 = np.prod(q1, dtype=np.longfloat)
        p1 = prod_1 ** (1 / k)
        # for output space
        prod_2 = np.prod(q2, dtype=np.longfloat)
        p2 = prod_2 ** (1 / k)
        # for both spaces - the geometric mean of the distortion in input space and output space
        prod_3 = np.prod(q1 * q2, dtype=np.longfloat)
        p3 = prod_3 ** (1 / (2 * k))

        # finally we store the topographic product components for the current unit in the result array.
        P[tuple(out_pos)] = p1, p2, p3

    # After calculating the topographic product components for each unit in the SOM,
    # we return the resulting 3D array, which can then be plotted as a heatmap (3 heatmaps actually).
    return P


if __name__ == "__main__":
    import time
    import seaborn as sns
    import matplotlib.pyplot as plt

    start_time = time.time()
    print("loading data...")
    chainlink_idata_path = os.path.join("datasets", "chainlink", "chainlink.vec")
    chainlink_idata = SOMToolBox_Parse(chainlink_idata_path).read_weight_file()
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("training...")
    m, n = 100, 60
    som = MiniSom(m, n, 3)
    som.train(chainlink_idata['arr'], 10000)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("calculating topographic product...")
    np.seterr(all='raise')
    tp = TopoProd(m, n, som._weights, k=8)
    N = m * n
    P = tp[..., 2].sum() / (N * (N - 1))
    print(f"Topographic product calculated in {time.time() - start_time:.2f} seconds")
    print("P =", P)

    # plot all components
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    sns.heatmap(tp[..., 0], ax=axs[0])
    axs[0].set_title('tp[...,0]')
    sns.heatmap(tp[..., 1], ax=axs[1])
    axs[1].set_title('tp[...,1]')
    sns.heatmap(tp[..., 2], ax=axs[2])
    axs[2].set_title('tp[...,2]')
    plt.show()