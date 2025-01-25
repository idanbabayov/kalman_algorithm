import pandas as pd
import numpy as np

def main():
    my_cols = ["A", "B", "C", "D", "E", "f", "g", "h", "i", "j", "k"]

    # Read the data and explicitly convert necessary columns to float64
    data = pd.read_csv("./src/kalman_algorithm/kalman_algorithm/data_radar_and_lidar.txt", 
                       names=my_cols, delim_whitespace=True, header=None)

    # Convert all numeric columns (except the first one) to float64
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(np.float64)

    # Check that all columns are now float64
    print(data.dtypes)

    # Initialize X_true_current as a float64 array
    X_true_current = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x_array = []

    firstMeasurement = data.iloc[0, :].values
    timeStamp = firstMeasurement[3]  # This is t_zero of our system

    # Make sure the precision is maintained
    np.set_printoptions(precision=15)  # Set high precision for printing

    for i in range(1, 5):
        currentMeas = data.iloc[i, :].values

        # Compute the current delta t
        if currentMeas[0] == 'L':
            # Ensure that all positions in X_true_current are updated
            X_true_current[0] = np.float64(currentMeas[1])  # Update first element (X position)
            X_true_current[1] = np.float64(currentMeas[2])  # Update second element (Y position)
            X_true_current[2] = np.float64(currentMeas[4])  # Update third element (vx )
            X_true_current[3] = np.float64(currentMeas[5])  # Update fourth element (vy)

        print(X_true_current)  # Print with full precision
        x_array.append(X_true_current.copy())  # Ensure we append a copy

    print(x_array)  # Final array with all values

if __name__ == '__main__':
    main()
