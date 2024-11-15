import numpy as np
from gym import spaces
import random
import scipy
import matplotlib.pyplot as plt
#from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise


class KF():
    def __init__(self):
        dt = 0.02

                # Define the state transition matrix
        transition_matrix = np.array([
        [1, 0, dt, 0, 0.5 * (dt ** 2), 0],
        [0, 1, 0, dt, 0, 0.5 * (dt ** 2)],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]])

        measurement_noise_covariance = np.eye(2)*1e-6  # A very small value

        observation_matrices = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        self.kf.x = np.array([0, 0, 0, 0, 0, 0])
        self.kf.F = transition_matrix
        self.kf.H = observation_matrices
        self.kf.R = measurement_noise_covariance
        self.kf.P *= 1e-3
        self.kf.Q *= 1e-3