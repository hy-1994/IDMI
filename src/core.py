import numba as nb
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


def D(X):
    X = np.array(X)
    X = X[X != 0]
    n = X
    N = np.sum(n)
    ans = np.sum(n * np.log(n))
    Dx = N * np.log(N) - ans
    return Dx

def Dif(X, Y):
    Dx = D(X)
    Dy = D(Y)
    XY = np.sum([X, Y], axis=0)
    Dxy = D(XY)
    dif = Dxy - Dx - Dy
    return dif

#metacell_assignments
@nb.jit
def metacell_assignments(ad, waypoint):
    data = ad.X.toarray()
    landmark_indices = waypoint 
    num_landmarks = len(landmark_indices)
    non_landmark_indices = [i for i in range(data.shape[0]) if i not in landmark_indices]
    num_non_landmarks = len(non_landmark_indices)
    cell_labels = ad.obs['metacell']


    for i in tqdm(non_landmark_indices, desc="Processing non-landmark cells", total=num_non_landmarks):
        min_similarity = float('inf')
        min_landmark_index = -1
        min_non_landmark_index = -1

        for j in landmark_indices:
            sim = Dif(data[i], data[j])
            if sim < min_similarity:
                min_similarity = sim
                min_landmark_index = j
                min_non_landmark_index = i

        cell_labels[min_non_landmark_index] = cell_labels[min_landmark_index]
        data[min_landmark_index] += data[min_non_landmark_index]
        num_non_landmarks -= 1

    return cell_labels