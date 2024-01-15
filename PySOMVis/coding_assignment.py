import numpy as np
import os
from SOMToolBox_Parse import SOMToolBox_Parse


def TopProd(_m, _n, _weights, _idata):  # noqa
    k = len(_weights)
    P3 = np.zeros((_m, _n))
    for i, w_j in enumerate(_weights):
        dist_inp_knn_out = ...
        dist_inp_knn_inp = ...
        dist_out_knn_out = ...
        dist_out_knn_inp = ...
        q1 = dist_inp_knn_out / dist_inp_knn_inp
        q2 = dist_out_knn_out / dist_out_knn_inp
        p3 = (q1 * q2) ** (1 / (2*k))
        P3[i // _n, i % _n] = p3

    return P3


if __name__ == "__main__":
    idata_path = os.path.join("datasets", "iris", "iris.vec")
    weights_path = os.path.join("datasets", "iris", "iris.wgt.gz")
    idata = SOMToolBox_Parse(idata_path).read_weight_file()
    weights = SOMToolBox_Parse(weights_path).read_weight_file()
    print(TopProd(weights['ydim'], weights['xdim'], weights['arr'], idata))
