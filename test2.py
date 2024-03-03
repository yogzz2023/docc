import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
import numpy as np
import csv

# Define the state transition matrix
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # Constant velocity model

# Define the measurement matrix
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Define the measurement noise covariance matrix
R = np.eye(2) * 5  # Assuming measurement noise with covariance 5

# Define initial state covariance
P = np.eye(4) * 100

# Define process noise covariance matrix
Q = np.eye(4) * 0.1  # Assuming process noise with covariance 0.1

def predict(x, P, Q, F):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, R, H):
    # Extract position for measurement
    x_pred_pos = x_pred[:2]
    
    # Ensure shapes are compatible for numpy operations
    if len(x_pred_pos) != len(H.T):
        print("Error: Incompatible shapes for numpy.dot in update()")
        return None, None
    
    y = z - np.dot(H, x_pred_pos)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    
    # Ensure S is invertible
    try:
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    except np.linalg.LinAlgError:
        print("Error: S is not invertible in update()")
        return None, None
    
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    
    return x_updated, P_updated

def process_data(measurements):
    global P  # Declare P as a global variable
    output_text.delete(1.0, tk.END)  # Clear previous output
    # Loop through each measurement
    for i, measurement in enumerate(measurements, start=1):
        output_text.insert(tk.END, f"\nProcessing Measurement {i}:\n")
        output_text.insert(tk.END, f"Measurement: {measurement}\n")
        # Predict
        x_pred, P_pred = predict(measurement, P, Q, F)
        output_text.insert(tk.END, f"Predicted Position: {x_pred[:2]}\n")
        print("Predicted Position:", x_pred[:2])  # Print predicted position
        # Update
        x_updated, P_updated = update(x_pred, P_pred, measurement, R, H)
        output_text.insert(tk.END, f"Updated Position: {x_updated[:2]}\n")
        print("Updated Position:", x_updated[:2])  # Print updated position

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                measurements = [np.array([float(val) for val in row]) for row in reader]
                process_data(measurements)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file: {e}")

# Create GUI
root = tk.Tk()
root.title("JPDA Kalman Filter")
root.configure(bg='white')

# Main frame with improved layout
main_frame = tk.Frame(root, bg='white')
main_frame.pack(padx=20, pady=20)

# Browse button
browse_button = tk.Button(main_frame, text="Browse CSV", font=('Helvetica', 12), bg='blue', fg='white', command=browse_file)
browse_button.grid(row=0, column=0, pady=10)

# Output frame
output_frame = tk.Frame(root, bg='white')
output_frame.pack(padx=20, pady=20)

# Output text area with improved appearance
output_text = scrolledtext.ScrolledText(output_frame, width=60, height=20, font=('Helvetica', 10), bg='black', fg='white')
output_text.pack()

root.mainloop()
