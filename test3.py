import numpy as np
from scipy.stats import multivariate_normal

class JPDAKalmanFilter:
    def __init__(self, F, H, Q, R, initial_state, initial_covariance):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.state = initial_state  # Initial state estimate
        self.covariance = initial_covariance  # Initial error covariance

    def predict(self):
        # Predict the next state
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, measurement):
        # Update the state estimate and covariance using Kalman gain
        innovation = measurement - np.dot(self.H, self.state).reshape(-1, 1)
        innovation_covariance = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        kalman_gain = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(innovation_covariance))
        self.state = self.state + np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, self.H), self.covariance)

    def jpda_update(self, measurements, association_probabilities):
        for i in range(len(measurements)):
            measurement = measurements[i]
            association_probability = association_probabilities[i]

            # Create temporary filter to handle association
            temp_filter = JPDAKalmanFilter(self.F, self.H, self.Q, self.R, self.state, self.covariance)
            temp_filter.update(measurement)

            # Ensure that temp_filter.covariance has the correct shape
            temp_filter.covariance = self.covariance

            # Update filter with association probability
            self.state = self.state + association_probability * (temp_filter.state - self.state)
            # Ensure covariance matrices have the same shape
            self.covariance = self.covariance + association_probability * np.outer((temp_filter.covariance - self.covariance).flatten(), 
                                                                                    (temp_filter.covariance - self.covariance).flatten()).reshape(self.covariance.shape)

def data_association(measurements, predicted_states, covariance_matrices, H, R):
    associations = []
    association_probabilities = []
    
    for i in range(len(predicted_states)):
        predicted_state = predicted_states[i]
        covariance = covariance_matrices[i]
        min_distance = float('inf')
        best_association = None
        
        for j in range(len(measurements)):
            measurement = measurements[j]
            # Transpose the predicted state for proper multiplication
            innovation = measurement - np.dot(H, predicted_state).reshape(-1, 1)
            innovation_covariance = np.dot(np.dot(H, covariance), H.T) + R
            mahalanobis_distance = np.dot(np.dot(innovation.T, np.linalg.inv(innovation_covariance)), innovation)
            
            # Compare each element of the array
            if np.all(mahalanobis_distance < min_distance):
                min_distance = mahalanobis_distance
                best_association = j
        
        association_probability = multivariate_normal.pdf(measurements[best_association], 
                                                          mean=np.dot(H, predicted_state).flatten(), 
                                                          cov=np.dot(np.dot(H, covariance), H.T) + R)
        
        associations.append(best_association)
        association_probabilities.append(association_probability)
    
    return associations, association_probabilities

def main():
    # Define system parameters
    dt = 1.0  # Time step
    F = np.array([[1, dt],
                  [0, 1]])  # State transition matrix
    H = np.array([[1, 0]])  # Measurement matrix
    Q = np.eye(2) * 0.01  # Process noise covariance
    R = np.eye(1) * 0.1  # Measurement noise covariance
    initial_state = np.array([[0],
                              [0]])  # Initial state estimate
    initial_covariance = np.eye(2) * 0.1  # Initial error covariance
    
    # Read measurements from text file
    with open('config.txt', 'r') as file:
        measurements = [list(map(float, line.strip().split(','))) for line in file]
    
    # Initialize JPDA Kalman filter
    jpda_kf = JPDAKalmanFilter(F, H, Q, R, initial_state, initial_covariance)
    
    # Main loop
    for measurement in measurements:
        # Perform prediction step
        jpda_kf.predict()
        
        # Perform data association
        associations, association_probabilities = data_association([measurement], [jpda_kf.state], [jpda_kf.covariance], H, R)
        
        # Perform JPDA update step
        jpda_kf.jpda_update([measurement], association_probabilities)
        
        # Print estimated state
        print("Estimated state:", jpda_kf.state.flatten())

if __name__ == "__main__":
    main()
