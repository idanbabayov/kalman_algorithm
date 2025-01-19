##
# Main function of the Python program.
#
##

import pandas as pd
import numpy as np



"""
#Fill in Those functions:

def computeRMSE(trueVector, EstimateVector):
    

    
def computeCovMatrix(deltaT, sigma_aX, sigma_aY):
??????????????????????????????????????
    

    
"""
def computeRadarJacobian(Xvector): # Jacobian matrix h with respect to state variables [Px, Py, Vx, Vy , yaw, yawRate]
#The Radar's readings are not linear when we convert them to our cartezian coordinate system.
#[rho,phi,rho_dot] = [sqrt(Px^2+Py^2), tg^-1(Py/Px),(Px*Vx+Py*Vy)/sqrt(Px^2+Py^2)].
#we need to develop a first order[linear] approximation around the mean[expected value], because in a normal disturbution this is where most of
# our values will be.(similar to a taylor series in 1 dimension around a point that will give best approximation)!

# Jacobian matrix h with respect to state variables [Px, Py, Vx, Vy , yaw, yawRate]
# Assuming h1(Px, Py, Vx, Vy) = rho =sqrt(Px^2+Py^2) , h2(Px, Py, Vx, Vy) = phi = tg^-1(Py/Px), and h3(Px, Py, Vx, Vy) = rho_dot =(Px*Vx+Py*Vy)/sqrt(Px^2+Py^2)
# h = [[ dh1/dPx  dh1/dPy  dh1/dVx  dh1/dVy dh1/dyaw dh1/dyawRate],
#      [ dh2/dPx  dh2/dPy  dh2/dVx  dh2/dVy dh2/dyaw dh2/dyawRate ],
#      [ dh3/dPx  dh3/dPy  dh3/dVx  dh3/dVy dh3/dyaw dh3/dyawRate ]]
#when solving we get:
# h = [[ d(sqrt(Px^2+Py^2))/dPx  d(sqrt(Px^2+Py^2))/dPy  d(sqrt(Px^2+Py^2))/dVx  d(sqrt(Px^2+Py^2))/dVy d(sqrt(Px^2+Py^2))/dyaw d(sqrt(Px^2+Py^2))/dyawRate],
#      [ d(tg^-1(Py/Px))/dPx  d(tg^-1(Py/Px))/dPy d(tg^-1(Py/Px))/dVx  d(tg^-1(Py/Px))/dVy d(tg^-1(Py/Px))/dyaw d(tg^-1(Py/Px))/dyawRate ],
#      [ d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dPx  d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dPy  d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dVx  d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dVy d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dyaw d((Px*Vx+Py*Vy)/sqrt(Px^2+Py^2))/dyawRate ]]
#
    Px = Xvector[0]
    Py = Xvector[1] 
    vx = Xvector[2] 
    vy = Xvector[3]
    yaw = Xvector[4]
    yawRate = Xvector[5]
    #for conviniecnce let us define :
    P_square =Px**2+Py**2 
    h_radar = [[ Px/(P_square)**0.5 , Py/(P_square)**0.5 , 0 , 0 , 0 ,0],
      [ -Px/(P_square) , Px/(P_square) , 0 , 0 , 0, 0 ],
      [ (Py*(vx*Py-vy*Px))/(P_square)**1.5  , (Px*(vy*Px-vx*Py))/(P_square)**1.5 , Px/(P_square)**0.5 , Py/(P_square)**0.5 , 0 , 0 ]] 


    return h_radar

def computeFmatrix(deltaT):
# Motion Model for a 2D Robot Vacuum Cleaner (Discrete Time) with a const linear speed.
# State vector: [Px, Py, Vx, Vy, Yaw, Yaw Rate]
# Px, Py: Positions in x and y
# Vx, Vy: Velocities in x and y
# Yaw: Robot's orientation (angle in radians)
# Yaw Rate: Rate of change of the yaw (angular velocity)


#Px, Py, Vx, Vy, Yaw, Yaw_rate = state vector x 
#The kinematic model equations are as follows:

#1)Px_new = Px + Vx * delta_t
#2)Py_new = Py + Vy * delta_t
#3)Vx_new = Vx
#4)Vy_new = Vy
#5)Yaw_new = Yaw + Yaw_rate * delta_t
#6)Yaw_rate_new = Yaw_rate

#Remark: deltaT is not constant

    F = np.matrix([ [1,0,deltaT,0,0,0],
                    [0,1,0,deltaT,0,0],
                    [0,0,1,0,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,deltaT],
                    [0,0,0,0,0,1] ])

    return F   
    
def main():
    # we print a heading and make it bigger using HTML formatting
    print("Hellow")
    my_cols = ["A", "B", "C", "D", "E","f","g","h","i","j","k"]
    data = pd.read_csv("./ros2_ws/src/kalman_algorithm/kalman_algorithm/data_radar_and_lidar.txt", names=my_cols, delim_whitespace = True, header=None)
    print(data.head())
    for i in range(10):
        measur = data.iloc[i,:].values
        #print(measur[0])

    #define matrices:   
    deltaT = 0.1 #known for an initial guess!
    F_matrix = computeFmatrix(deltaT)
    # Covariance matrix (Sigma)
# [
#     [sigma_px^2, sigma_px_py, sigma_px_vx, sigma_px_vy, sigma_px_yaw, sigma_px_yawRate],  # Covariance between position x and other variables
#     [sigma_py_px, sigma_py^2, sigma_py_vx, sigma_py_vy, sigma_py_yaw, sigma_py_yawRate],  # Covariance between position y and other variables
#     [sigma_vx_px, sigma_vx_py, sigma_vx^2, sigma_vx_vy, sigma_vx_yaw, sigma_vx_yawRate],  # Covariance between velocity x and other variables
#     [sigma_vy_px, sigma_vy_py, sigma_vy_vx, sigma_vy^2, sigma_vy_yaw, sigma_vy_yawRate],   # Covariance between velocity y and other variables
#     [sigma_yaw_px,sigma_yaw_py, sigma_yaw_vx, sigma_yaw_vy, sigma_yaw^2, sigma_yaw_yawRate ], # Covariance between yaw and other variables
#     [sigma_yaw_px,sigma_yaw_py, sigma_yaw_vx, sigma_yaw_vy, sigma_yaw_yawRate, sigma_yaw_yawRate^2 ]  # Covariance between yawRate and other variables
# ]
 
# Example of a possible P matrix: Initial state covariance matrix P (6x6) with covariance between position and velocity
# It is an initial guess of the uncertainty of the inital guess of the state.

    P = np.array([
        [4.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # High uncertainty in Px, small covariance with Vx
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # Small covariance between Px and Vx
        [0.0, 0.0, 4.0, 1.0, 0.0, 0.0],  # High uncertainty in Py, small covariance with Vy
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],   # Small covariance between Py and Vy
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],   # Low uncertainty in yaw
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]   # Low uncertainty in yaw rate
    ])

    H_Lidar = np.matrix([[1,0,0,0,0,0],
                        [0,1,0,0,0,0]])
    R_lidar = np.array([[0.0225, 0.0],
                        [0.0, 0.0225]]) #known
    R_radar = np.array([[0.9, 0, 0],
                        [0.0, 0.0009, 0],
                        [0, 0, 0.09]])  #known
    useRadar = False
    xEstimate = []
    xTrue = []  
    #fill in X_true and X_state. Put 0 for the velocities
    X_state_current = [1.2,1.2,0.0,0.0,0.2,0.03] #initial state guess
    X_true_current = [2.0,2.4,0.0,0.0,0.1,0.06]  #initial state true!(suppose we know it)

    firstMeasurment = data.iloc[0,:].values
    timeStamp = firstMeasurment[3]
    
    for index in range(1,len(data)):
        currentMeas = data.iloc[i,:].values

        # compute the current dela t
        if(currentMeas[0]=='L'):
            
            deltaT = (currentMeas[3]- timeStamp)/1000000
            timeStamp = currentMeas[3]
            
            #perfrom predict

            X_state_current =  F_matrix * X_state_current 
            P  = F_matrix * P * F_matrix.transpose()

            #pefrom measurment update
            z = 
            S = 
            K = 
            X_state_current = 
            P  = 
            


    """
            
        if(currentMeas[0]=='R' and useRadar):
            
             #perfrom predict
            deltaT = (currentMeas[4]- timeStamp)/1000000
            timeStamp = currentMeas[4]
            X_state_current = 
            P  = 
            
            #pefrom measurment update
            jacobian = computeRadarJacobian(X_state_current)
            z = 
            S = 
            K = 
            X_state_current = 
            P  = 
            
            
            
        xEstimate.append(X_state_current)
        xTrue.append(X_true_current)
            
    rmse = computeRmse(xEstimate, xTrue) 
    print(rmse)

    """
        
    

if __name__ == '__main__':
    main()
