import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
import numpy as np

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

def predict(x, P, Q, F):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, R, H):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated

def process_data():
    global P  # Declare P as a global variable
    output_text.delete(1.0, tk.END)  # Clear previous output
    # Get initial target positions from input fields
    try:
        target1 = np.array([float(entry_target1_x.get()), float(entry_target1_y.get()), 0, 0])
        target2 = np.array([float(entry_target2_x.get()), float(entry_target2_y.get()), 0, 0])

        # Range, azimuth, elevation, and time
        range_1 = float(entry_range_1.get())
        azimuth_1 = float(entry_azimuth_1.get())
        elevation_1 = float(entry_elevation_1.get())
        time_1 = float(entry_time_1.get())

        range_2 = float(entry_range_2.get())
        azimuth_2 = float(entry_azimuth_2.get())
        elevation_2 = float(entry_elevation_2.get())
        time_2 = float(entry_time_2.get())

        # Convert range, azimuth, elevation to x, y coordinates
        m1 = np.array([range_1 * np.cos(azimuth_1), range_1 * np.sin(azimuth_1)])
        m2 = np.array([range_2 * np.cos(azimuth_2), range_2 * np.sin(azimuth_2)])

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
            for j, measurement in enumerate([m1, m2], start=1):
                x_pred, P_pred = predict(target, P, Q, F)
                x_updated, P_updated = update(x_pred, P_pred, measurement, R, H)
                association_prob = P_M[i-1, j-1]
                target[:4] = target[:4] + association_prob * (x_updated[:4] - target[:4])
                output_text.insert(tk.END, f"Measurement {j}: {measurement}\n")
                output_text.insert(tk.END, f"Association Probability: {association_prob}\n")
                output_text.insert(tk.END, f"Updated Position: {target[:2]}\n")

            output_text.insert(tk.END, f"Final Updated Position: {target[:2]}\n")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

def save_output():
    text = output_text.get(1.0, tk.END)
    with open("output_data.txt", "w") as file:
        file.write(text)

def calculate_predicted_position():
    global P
    try:
        target1 = np.array([float(entry_target1_x.get()), float(entry_target1_y.get()), 0, 0])
        target2 = np.array([float(entry_target2_x.get()), float(entry_target2_y.get()), 0, 0])
        
        target1_pred, _ = predict(target1, P, Q, F)
        target2_pred, _ = predict(target2, P, Q, F)
        
        output_text.insert(tk.END, f"\nPredicted Position for Target 1: {target1_pred[:2]}\n")
        output_text.insert(tk.END, f"Predicted Position for Target 2: {target2_pred[:2]}\n")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

# Create GUI
root = tk.Tk()
root.title("JPDA Kalman Filter")
root.configure(bg='white')

# Save Output button
save_button = tk.Button(root, text="Save Output", font=('Helvetica', 12), bg='red', fg='white', command=save_output)
save_button.pack(pady=10)

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

tk.Label(main_frame, text="Measurement 1:", font=('Helvetica', 12), bg='white', fg='black').grid(row=8, column=0, sticky="w", pady=5)
tk.Label(main_frame, text="Range:", font=('Helvetica', 10), bg='white', fg='black').grid(row=9, column=0, sticky="w")
tk.Label(main_frame, text="Azimuth:", font=('Helvetica', 10), bg='white', fg='black').grid(row=10, column=0, sticky="w")
tk.Label(main_frame, text="Elevation:", font=('Helvetica', 10), bg='white', fg='black').grid(row=11, column=0, sticky="w")
tk.Label(main_frame, text="Time (ms):", font=('Helvetica', 10), bg='white', fg='black').grid(row=12, column=0, sticky="w")

tk.Label(main_frame, text="Measurement 2:", font=('Helvetica', 12), bg='white', fg='black').grid(row=8, column=2, sticky="w", pady=5)
tk.Label(main_frame, text="Range:", font=('Helvetica', 10), bg='white', fg='black').grid(row=9, column=2, sticky="w")
tk.Label(main_frame, text="Azimuth:", font=('Helvetica', 10), bg='white', fg='black').grid(row=10, column=2, sticky="w")
tk.Label(main_frame, text="Elevation:", font=('Helvetica', 10), bg='white', fg='black').grid(row=11, column=2, sticky="w")
tk.Label(main_frame, text="Time (ms):", font=('Helvetica', 10), bg='white', fg='black').grid(row=12, column=2, sticky="w")

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

entry_range_1 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_range_1.grid(row=9, column=1)
entry_azimuth_1 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_azimuth_1.grid(row=10, column=1)
entry_elevation_1 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_elevation_1.grid(row=11, column=1)
entry_time_1 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_time_1.grid(row=12, column=1)

entry_range_2 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_range_2.grid(row=9, column=3)
entry_azimuth_2 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_azimuth_2.grid(row=10, column=3)
entry_elevation_2 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_elevation_2.grid(row=11, column=3)
entry_time_2 = tk.Entry(main_frame, font=('Helvetica', 10), bg='gray90')
entry_time_2.grid(row=12, column=3)

# Buttons with improved appearance and layout
button_frame = tk.Frame(main_frame, bg='white')
button_frame.grid(row=13, columnspan=4, pady=10)

predict_button = tk.Button(button_frame, text="Calculate Predicted Position", font=('Helvetica', 12), bg='green', fg='white', command=calculate_predicted_position)
predict_button.pack(side=tk.LEFT, padx=10)

association_button = tk.Button(button_frame, text="Calculate Association Probability", font=('Helvetica', 12), bg='blue', fg='white', command=process_data)
association_button.pack(side=tk.LEFT, padx=10)

update_button = tk.Button(button_frame, text="Calculate Updated Position", font=('Helvetica', 12), bg='orange', fg='white', command=process_data)
update_button.pack(side=tk.LEFT, padx=10)

# Output frame
output_frame = tk.Frame(root, bg='white')
output_frame.pack(padx=20, pady=20)

# Output text area with improved appearance
output_text = scrolledtext.ScrolledText(output_frame, width=60, height=20, font=('Helvetica', 10), bg='black', fg='white')
output_text.pack()

root.mainloop()
