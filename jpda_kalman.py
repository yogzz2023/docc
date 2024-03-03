import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
import numpy as np

class JPDAKalmanFilterGUI:
    def __init__(self, root):
        self.root = root
        root.title("JPDA Kalman Filter")
        root.configure(bg='white')

        self.create_input_widgets()
        self.create_buttons()
        self.create_output_frame()

        # Kalman Filter Parameters
        self.F = np.array([[1, 0, 0.001, 0],
                           [0, 1, 0, 0.001],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # Constant velocity model
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 5  # Measurement noise covariance matrix
        self.Q = np.eye(4) * 0.1  # Process noise covariance matrix
        self.P = np.eye(4) * 100  # Initial state covariance

    def create_input_widgets(self):
        self.main_frame = tk.Frame(self.root, bg='white')
        self.main_frame.pack(padx=20, pady=20)

        labels = ["Target 1 (x, y):", "Target 2 (x, y):", "Pd (Prob. of Detection):", "Pfa (Prob. of False Alarm):"]
        row_positions = [0, 3, 6, 8]

        for i, label_text in enumerate(labels):
            tk.Label(self.main_frame, text=label_text, font=('Helvetica', 12), bg='white', fg='black').grid(row=row_positions[i], column=0, sticky="w", pady=5)

        self.entry_fields = {}
        entry_labels = ["target1_x", "target1_y", "target2_x", "target2_y", "Pd", "Pfa"]
        for i, label in enumerate(entry_labels):
            self.entry_fields[label] = tk.Entry(self.main_frame, font=('Helvetica', 10), bg='gray90')
            self.entry_fields[label].grid(row=i // 2, column=i % 2 + 1)

    def create_buttons(self):
        button_frame = tk.Frame(self.main_frame, bg='white')
        button_frame.grid(row=9, columnspan=3, pady=10)

        buttons = [("Calculate Predicted Position", self.calculate_predicted_position),
                   ("Calculate Association Probability", self.process_data),
                   ("Calculate Updated Position", self.process_data)]

        for button_text, command in buttons:
            button = tk.Button(button_frame, text=button_text, font=('Helvetica', 12), command=command)
            button.pack(side=tk.LEFT, padx=10)

    def create_output_frame(self):
        output_frame = tk.Frame(self.root, bg='white')
        output_frame.pack(padx=20, pady=20)

        self.output_text = scrolledtext.ScrolledText(output_frame, width=60, height=20, font=('Helvetica', 10), bg='black', fg='white')
        self.output_text.pack()

    def calculate_predicted_position(self):
        try:
            target1 = np.array([float(self.entry_fields["target1_x"].get()), float(self.entry_fields["target1_y"].get()), 0, 0])
            target2 = np.array([float(self.entry_fields["target2_x"].get()), float(self.entry_fields["target2_y"].get()), 0, 0])

            target1_pred, _ = self.predict(target1, self.P, self.Q, self.F)
            target2_pred, _ = self.predict(target2, self.P, self.Q, self.F)

            self.output_text.insert(tk.END, f"\nPredicted Position for Target 1: {target1_pred[:2]}\n")
            self.output_text.insert(tk.END, f"Predicted Position for Target 2: {target2_pred[:2]}\n")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

    def process_data(self):
        self.output_text.delete(1.0, tk.END)  # Clear previous output
        try:
            # Get target positions and parameters
            target1 = np.array([float(self.entry_fields["target1_x"].get()), float(self.entry_fields["target1_y"].get()), 0, 0])
            target2 = np.array([float(self.entry_fields["target2_x"].get()), float(self.entry_fields["target2_y"].get()), 0, 0])
            Pd = float(self.entry_fields["Pd"].get())
            Pfa = float(self.entry_fields["Pfa"].get())

            # Placeholder for measurements
            measurements = []
            for i in range(1, 3):
                range_val = float(input(f"Enter range for Measurement {i}: "))
                azimuth_val = float(input(f"Enter azimuth for Measurement {i}: "))
                elevation_val = float(input(f"Enter elevation for Measurement {i}: "))
                time_val = float(input(f"Enter time for Measurement {i} (ms): "))

                # Convert range, azimuth, elevation to x, y coordinates
                x_coord = range_val * np.cos(azimuth_val)
                y_coord = range_val * np.sin(azimuth_val)
                measurements.append(np.array([x_coord, y_coord]))

            # Calculate association probabilities
            P_M = np.array([[Pd * (1 - Pfa), Pd * Pfa],
                            [(1 - Pd) * (1 - Pfa), (1 - Pd) * Pfa]])

            # Loop through each target
            for i, target in enumerate([target1, target2], start=1):
                self.output_text.insert(tk.END, f"\nProcessing Target {i}:\n")
                self.output_text.insert(tk.END, f"Initial Position: {target[:2]}\n")

                # Predict
                target[:4], self.P = self.predict(target[:4], self.P, self.Q, self.F)
                self.output_text.insert(tk.END, f"Predicted Position: {target[:2]}\n")

                # Update
                for j, measurement in enumerate(measurements, start=1):
                    x_pred, P_pred = self.predict(target, self.P, self.Q, self.F)
                    x_updated, P_updated = self.update(x_pred, P_pred, measurement, self.R, self.H)
                    association_prob = P_M[i-1, j-1]
                    target[:4] = target[:4] + association_prob * (x_updated[:4] - target[:4])
                    self.output_text.insert(tk.END, f"Measurement {j}: {measurement}\n")
                    self.output_text.insert(tk.END, f"Association Probability: {association_prob}\n")
                    self.output_text.insert(tk.END, f"Updated Position: {target[:2]}\n")

                self.output_text.insert(tk.END, f"Final Updated Position: {target[:2]}\n")

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

    def predict(self, x, P, Q, F):
        x_pred = np.dot(F, x)
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z, R, H):
        y = z - np.dot(H, x_pred)
        S = np.dot(np.dot(H, P_pred), H.T) + R
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
        x_updated = x_pred + np.dot(K, y)
        P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
        return x_updated, P_updated

if __name__ == "__main__":
    root = tk.Tk()
    app = JPDAKalmanFilterGUI(root)
    root.mainloop()
