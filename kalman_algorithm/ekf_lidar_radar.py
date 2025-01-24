##
# Main function of the Python program.
#
##

import pandas as pd
import numpy as np



def computeCovMatrix(deltaT, sigma_aX, sigma_aY):
        # Covariance matrix (Sigma)
# [
#     [sigma_px^2, sigma_px_py, sigma_px_vx, sigma_px_vy],  # Covariance between position x and other variables
#     [sigma_py_px, sigma_py^2, sigma_py_vx, sigma_py_vy],  # Covariance between position y and other variables
#     [sigma_vx_px, sigma_vx_py, sigma_vx^2, sigma_vx_vy,  # Covariance between velocity x and other variables
#     [sigma_vy_px, sigma_vy_py, sigma_vy_vx, sigma_vy^2],   # Covariance between velocity y and other variables
# ]
 
# Example of a possible P matrix: Initial state covariance matrix P (6x6) with covariance between position and velocity
# It is an initial guess of the uncertainty of the inital guess of the state.


    cov = np.eye(4)  # Start with an identity matrix (or any other initial guess)
    cov *= 1000      # Scale it to represent high initial uncertainty
    return cov

def computeRmse(trueVector, EstimateVector):
    # Ensure that both trueVector and EstimateVector are NumPy arrays for element-wise operations
    trueVector = np.array(trueVector)
    EstimateVector = np.array(EstimateVector)

    # Check if the shapes of the true and estimate vectors are compatible
    if trueVector.shape != EstimateVector.shape:
        raise ValueError(f"Input vectors must have the same shape. Got {trueVector.shape} and {EstimateVector.shape}.")

    # Compute squared differences
    squared_diff = (trueVector - EstimateVector) ** 2

    # Compute the mean squared error for each state component (across all measurements)
    # Compute the mean across axis 0 (columns)
    mse = np.mean(squared_diff, axis=0)

    # Compute the RMSE by taking the square root of the mean squared error
    rmse = np.sqrt(mse)

    return rmse

def computeRadarJacobian(Xvector): # Jacobian matrix h with respect to state variables [Px, Py, Vx, Vy ]
#The Radar's readings are not linear when we convert them to our cartezian coordinate system.
#[rho,phi,rho_dot] = [sqrt(Px^2+Py^2), tg^-1(Py/Px),(Px*Vx+Py*Vy)/sqrt(Px^2+Py^2)].
#we need to develop a first order[linear] approximation around the mean[expected value], because in a normal disturbution this is where most of
# our values will be.(similar to a taylor series in 1 dimension around a point that will give best approximation)!

# Jacobian matrix h with respect to state variables [Px, Py, Vx, Vy , yaw, yawRate]
# Assuming h1(Px, Py, Vx, Vy) = rho =sqrt(Px^2+Py^2) , h2(Px, Py, Vx, Vy) = phi = tg^-1(Py/Px), and h3(Px, Py, Vx, Vy) = rho_dot =(Px*Vx+Py*Vy)/sqrt(Px^2+Py^2)
# h = [[ dh1/dPx  dh1/dPy  dh1/dVx  dh1/dVy ],
#      [ dh2/dPx  dh2/dPy  dh2/dVx  dh2/dVy  ],
#      [ dh3/dPx  dh3/dPy  dh3/dVx  dh3/dVy ]]
#when solving we get:
# h = [[ d(sqrt(Px^2+Py^2))/dPx  d(sqrt(Px^2+Py^2))/dPy  d(sqrt(Px^2+Py^2))/dVx  d(sqrt(Px^2+Py^2))/dVy ],
#      [ d(tg^-1(Py/Px))/dPx  d(tg^-1(Py/Px))/dPy d(tg^-1(Py/Px))/dVx  d(tg^-1(Py/Px))/dVy  ],
#      [ d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dPx  d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dPy  d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dVx  d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dVy  ]]
#
    # Assuming Xvector is a 2D column vector of shape (4, 1) [Px, Py, Vx, Vy]
    Px = Xvector[0, 0]  # Extract the scalar value from the 2D column vector
    Py = Xvector[1, 0]  # Extract the scalar value from the 2D column vector
    vx = Xvector[2, 0]  # Extract the scalar value from the 2D column vector
    vy = Xvector[3, 0]  # Extract the scalar value from the 2D column vector

    # For convenience, define:
    P_square = Px**2 + Py**2 
    h_radar = np.array([
        [ Px / np.sqrt(P_square),    Py / np.sqrt(P_square),    0,    0 ],
        [ -Py / P_square,            Px / P_square,             0,    0 ],
        [ (Py * (vx * Py - vy * Px)) / np.power(P_square, 1.5),
          (Px * (vy * Px - vx * Py)) / np.power(P_square, 1.5),
          Px / np.sqrt(P_square),    Py / np.sqrt(P_square) ]
    ])

    return h_radar

def computeFmatrix(deltaT):
# Motion Model for a 2D Robot Vacuum Cleaner (Discrete Time) with a const linear speed.
# State vector: [Px, Py, Vx, Vy, Yaw, Yaw Rate]
# Px, Py: Positions in x and y
# Vx, Vy: Velocities in x and y


#Px, Py, Vx, Vy = state vector x 
#The kinematic model equations are as follows:

#1)Px_new = Px + Vx * delta_t
#2)Py_new = Py + Vy * delta_t
#3)Vx_new = Vx
#4)Vy_new = Vy
#Remark: deltaT is not constant

    F = np.matrix([ [1,0,deltaT,0],
                    [0,1,0,deltaT],
                    [0,0,1,0],
                    [0,0,0,1]])

    return F   
    
def main():
    my_cols = ["A", "B", "C", "D", "E","f","g","h","i","j","k"]
    data = pd.read_csv("./src/kalman_algorithm/kalman_algorithm/data_radar_and_lidar.txt", names=my_cols, delim_whitespace = True, header=None)
    print(data.head())

    #define matrices:   
    deltaT = 0.1#known for an initial guess!

    P = computeCovMatrix(deltaT, 1.0, 1.0)

    H_Lidar = np.matrix([[1,0,0,0],
                        [0,1,0,0]])
    R_lidar = np.array([[0.0225, 0.0],
                        [0.0, 0.0225]]) #known
    R_radar = np.array([[0.9, 0, 0],
                        [0.0, 0.0009, 0],
                        [0, 0, 0.09]])  #known
    useRadar = True
    xEstimate = []
    xTrue = []  
    #fill in X_true and X_state. Put 0 for the velocities
    X_state_current = np.array([2.2,1.2,0.0,0.0]) #initial state guess
    X_true_current = np.array([0.0,0.0,0.0,0.0]) #initial empty

    firstMeasurment = data.iloc[0,:].values
    timeStamp = firstMeasurment[3] #this is t_zero of out system
    
    for i in range(1,len(data)):
        currentMeas = data.iloc[i,:].values

        # compute the current delta t
        if(currentMeas[0]=='L'):
            for j in range(4,len(X_true_current)):
                X_true_current[j-4] = currentMeas[j] #for each measurement we can take from the table the true value of the specified measurement.

            deltaT = (currentMeas[3]- timeStamp)/1000000
            timeStamp = currentMeas[3]
           
            #perfrom predict
            if deltaT > 0:
                F_matrix = computeFmatrix(deltaT) # the prediction is for each time we got a measurment update from sensor
                X_state_current =  F_matrix * X_state_current.reshape(-1,1)
                P  = F_matrix * P * F_matrix.transpose()
            if deltaT == 0: #the predictoin was already done for this deltaT between states.
                print("dt=0 and the time is:",timeStamp,"and the line is",i)

            #pefrom measurment update
            z = np.matrix([[currentMeas[1]],
                        [currentMeas[2]]])  # Convert Px and Py measurement to column vector
            y = H_Lidar * X_state_current #mu(best estimate) of the predicted state converted into the measurement[Lidar] space!
            S = H_Lidar * P * H_Lidar.transpose() #sigma (error) of the predicted state converted into the measurement space!
            K = P * H_Lidar.transpose()*np.linalg.inv(S + R_lidar) #SEE THAT IT IS THE SAME! P * H.transpose()*np.linalg.inv(H * P * H.transpose()+ R)

            X_state_current = X_state_current + K*(z-y)  #SEE THAT IT IS THE SAME! x + K * (Z - H * x ))

            P = (np.eye(4) - (K * H_Lidar)) * P # this is simplification of P - K * H * P 

        if(currentMeas[0]=='R' and useRadar):

            deltaT = (currentMeas[4]- timeStamp)/1000000
            timeStamp = currentMeas[4]
             #perfrom predict

            if deltaT > 0:
                F_matrix = computeFmatrix(deltaT) # the prediction is for each time we got a measurment update from sensor
                X_state_current =  F_matrix * X_state_current.reshape(-1,1)
                P  = F_matrix * P * F_matrix.transpose()
            if deltaT == 0: #the predictoin was already done for this deltaT between states.
                print("dt=0 and the time is:",timeStamp,"and the line is",i)

            
            #pefrom measurment update
            jacobian = computeRadarJacobian(X_state_current) # In this case the jacobian is an approximation of transformation from 
            #the prediction state space into the measurement space, because the measurement space is not linear for Px,Py,vx,vy.
            z = np.matrix([[currentMeas[1]],
                        [currentMeas[2]],
                        [currentMeas[3]]])  # Convert rho,phi,rho_dot to column vector
            y = jacobian * X_state_current #mu(best estimate) of the predicted state converted into the measurement[Radar] space!
            S = jacobian * P * jacobian.transpose()  #sigma (uncertainty) of the predicted state converted into the measurement space!
            K = P * jacobian.transpose()*np.linalg.inv(S+R_radar)
            X_state_current = X_state_current + K*(z - y)
            P = (np.eye(4) - (K * jacobian)) * P # this is simplification of P - K * H * P 
            
                        
        xEstimate.append(X_state_current) 
        xTrue.append(X_true_current.reshape(-1,1))
    print(np.shape(xTrue))  
    print(np.shape(xEstimate)) 
    rmse = computeRmse(xEstimate, xTrue) 
    print(rmse)
            



if __name__ == '__main__':
    main()
