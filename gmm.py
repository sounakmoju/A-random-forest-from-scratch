import math
import numpy as np
from math import pi

def euclidean_dist(pointA, pointB):
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)
def gaussian_kernal(distance,bandwidth):
    euclidean_distance=np.sqrt(((distance)**2).sum(axis=1))
    val=(1/(bandwidth.math.sqrt(2*math.pi)))*np.exp(-0.5*((euclidean_distance/bandwidth)))

    return val
def multivariate_gaussian_kernal(distances,bandwidths):
    dim=len(bandwidths)
    cov=np.multiply(np.power(bandwidths,2),np.eye(dim))
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val=(1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)
    return val
