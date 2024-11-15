
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def compute_DVH(dose, mask, maxDVH, number_of_bins):
    DVH_interval = [0, maxDVH + 2]
    bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
    bin_edges = np.arange(DVH_interval[0], DVH_interval[1], bin_size)
    bin_edges_interpolate = np.arange(DVH_interval[0], DVH_interval[1], bin_size/10)
    d = dose[mask!=0]
    dvh = np.zeros(len(bin_edges))
    for i in range(len(dvh)) :
        dvh[i] = len(d[d>=bin_edges[i]])*100/len(d)

    return bin_edges,dvh, bin_edges_interpolate

def compute_DVH_OAR(dose, mask, maxDVH, number_of_bins):
    DVH_interval = [0, maxDVH + 2]
    bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
    bin_edges = np.arange(DVH_interval[0], DVH_interval[1], bin_size)
    bin_edges_interpolate = np.arange(DVH_interval[0], DVH_interval[1], bin_size/10)
    d = dose[mask==0]
    dvh = np.zeros(len(bin_edges))
    for i in range(len(dvh)) :
        dvh[i] = len(d[d>=bin_edges[i]])*100/len(d)

    return bin_edges,dvh, bin_edges_interpolate

def computeDx(d, percentile, maxDVH):
    number_of_bins = 4096
    DVH_interval = [0, maxDVH]
    bin_size = (DVH_interval[1] - DVH_interval[0]) / number_of_bins
    bin_edges = np.arange(DVH_interval[0], DVH_interval[1] + 0.5 * bin_size, bin_size)
    bin_edges[-1] = maxDVH + d.max()
    _dose = bin_edges[:number_of_bins] + 0.5 * bin_size
    h, _ = np.histogram(d, bin_edges)
    h = np.flip(h, 0)
    h = np.cumsum(h)
    h = np.flip(h, 0)
    _volume = h * 100 / len(d)  # volume in %
    index = np.searchsorted(-_volume, -percentile)
    if (index > len(_volume) - 2): 
        index = len(_volume) - 2
    volume = _volume[index]
    volume2 = _volume[index + 1]
    if (volume == volume2):
            Dx = _dose[index]
    else:
            w2 = (volume - percentile) / (volume - volume2)
            w1 = (percentile - volume2) / (volume - volume2)
            Dx = w1 * _dose[index] + w2 * _dose[index + 1]
            if Dx < 0: Dx = 0
    return Dx