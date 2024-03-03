import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Define your Kalman Filter
def create_kalman_filter():
    # Define Kalman Filter parameters
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # Define state transition matrix
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # Define measurement function
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    # Define measurement noise covariance
    kf.R = np.eye(2) * 0.1
    # Define process noise covariance
    kf.Q = np.eye(4) * 0.01
    # Define initial state covariance
    kf.P = np.eye(4) * 500
    # Define initial state estimate
    kf.x = np.array([[0, 0, 0, 0]]).T
    return kf

# JPDA Kalman Filter function
def jpda_kalman_filter(measurements):
    kf = create_kalman_filter()
    # Perform JPDA association
    for z in measurements:
        probabilities = []  # List to store probabilities
        for j in range(len(kf.x)):
            kf.predict()
            kf.update(z)
            probabilities.append(kf.log_likelihood)
        # Perform assignment using Hungarian algorithm
        probabilities = np.array(probabilities)
        association = linear_sum_assignment(probabilities)
        # Update Kalman Filter with assigned measurements
        for j, z_index in enumerate(association[1]):
            kf.update(measurements[z_index])
    return kf.x

# Example usage
measurements = [[1, 2], [3, 4], [5, 6]]  # Example measurements
result = jpda_kalman_filter(measurements)
print("Result:", result)
