import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import math

# Define the state transition matrix
F = np.array([[1, 0, 0.001, 0],
              [0, 1, 0, 0.001],
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

# Function to predict the next state
def predict(x, P, Q, F):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

# Function to update the state based on measurements
def update(x_pred, P_pred, z, R, H):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated

# Function to load data from a config file
def load_config():
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filename:
        with open(filename, 'r') as file:
            data = file.readlines()[1:]  # Skip the header line
            measurements = [list(map(float, line.strip().split(','))) for line in data]
            process_measurements(measurements)
            
            
# Process loaded measurements and perform calculations
def process_measurements(measurements):
    global P  # Declare P as a global variable
    output_text.delete(1.0, tk.END)  # Clear previous output
    # Get initial target positions from input fields
    try:
        target1 = np.array([float(entry_target1_x.get()), float(entry_target1_y.get()), 0, 0])
        target2 = np.array([float(entry_target2_x.get()), float(entry_target2_y.get()), 0, 0])

        # Association probabilities
        Pd = float(entry_Pd.get())
        Pfa = float(entry_Pfa.get())

        # Calculate association probabilities
        P_M = np.array([[Pd * (1 - Pfa), Pd * Pfa],
                        [(1 - Pd) * (1 - Pfa), (1 - Pd) * Pfa]])

        # Loop through each target
        for i, target in enumerate([target1, target2], start=1):
            output_text.insert(tk.END, f"\nProcessing Target {i}:\n")
            output_text.insert(tk.END, f"Initial Position: {target[:2]}\n")

            # Predict
            target[:4], P = predict(target[:4], P, Q, F)
            output_text.insert(tk.END, f"Predicted Position: {target[:2]}\n")

            # Update
            j = 0
            while j < len(measurements):
                x_pred, P_pred = predict(target, P, Q, F)
                measurement = measurements[j]
                measurement_xy = polar_to_xy(measurement[0], math.radians(measurement[1]))
                x_updated, P_updated = update(x_pred, P_pred, measurement_xy, R, H)
                association_prob = P_M[i-1, j % P_M.shape[1]]  # Cycle through P_M if needed
                target[:4] = target[:4] + association_prob * (x_updated[:4] - target[:4])
                output_text.insert(tk.END, f"Measurement {j+1}: {measurement_xy}\n")
                output_text.insert(tk.END, f"Association Probability: {association_prob}\n")
                output_text.insert(tk.END, f"Updated Position: {target[:2]}\n")
                j += 1

            output_text.insert(tk.END, f"Final Updated Position: {target[:2]}\n")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values.")


# Convert polar coordinates (range, azimuth) to Cartesian coordinates (x, y)
def polar_to_xy(range_, azimuth):
    x = range_ * np.cos(azimuth)
    y = range_ * np.sin(azimuth)
    return np.array([x, y])

# Create GUI
root = tk.Tk()
root.title("JPDA Kalman Filter")
root.configure(bg='white')

# Main frame with improved layout
main_frame = tk.Frame(root, bg='white')
main_frame.pack(padx=20, pady=20)

# Labels
tk.Label(main_frame, text="Target 1 (x, y):", font=('Helvetica', 12), bg='white', fg='black').grid(row=0, column=0, sticky="w", pady=5)
tk.Label(main_frame, text="x:", font=('Helvetica', 10), bg='white', fg='black').grid(row=1, column=0, sticky="w")
tk.Label(main_frame, text="y:", font=('Helvetica', 10), bg='white', fg='black').grid(row=2, column=0, sticky="w")

tk.Label(main_frame, text="Target 2 (x, y):", font=('Helvetica', 12), bg='white', fg='black').grid(row=3, column=0, sticky="w", pady=5)
tk.Label(main_frame, text="x:", font=('Helvetica', 10), bg='white', fg='black').grid(row=4, column=0, sticky="w")
tk.Label(main_frame, text="y:", font=('Helvetica', 10), bg='white', fg='black').grid(row=5, column=0, sticky="w")

tk.Label(main_frame, text="Pd (Prob. of Detection):", font=('Helvetica', 12), bg='white', fg='black').grid(row=6, column=0, sticky="w", pady=5)
tk.Label(main_frame, text="Pfa (Prob. of False Alarm):", font=('Helvetica', 12), bg='white', fg='black').grid(row=7, column=0, sticky="w", pady=5)

# Entry fields
entry_target1_x = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_target1_x.grid(row=1, column=1)
entry_target1_y = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_target1_y.grid(row=2, column=1)

entry_target2_x = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_target2_x.grid(row=4, column=1)
entry_target2_y = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_target2_y.grid(row=5, column=1)

entry_Pd = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_Pd.grid(row=6, column=1)
entry_Pfa = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_Pfa.grid(row=7, column=1)

# Load Config button
load_button = tk.Button(main_frame, text="Load Config", font=('Helvetica', 12), bg='blue', fg='white', command=load_config)
load_button.grid(row=8, column=1, sticky="w", padx=10, pady=20)

# Process Data button
process_button = tk.Button(main_frame, text="Process Data", font=('Helvetica', 12), bg='green', fg='white', command=process_measurements)
process_button.grid(row=9, column=0, columnspan=2, pady=10)

# Output frame
output_frame = tk.Frame(root, bg='white')
output_frame.pack(padx=20, pady=20)

# Output text area with improved appearance
output_text = scrolledtext.ScrolledText(output_frame, width=60, height=20, font=('Helvetica', 10), bg='black', fg='white')
output_text.pack()

root.mainloop()
