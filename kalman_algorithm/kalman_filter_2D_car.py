##
# Main function of the Python program.
#
##
import numpy as np
from math import *
from numpy.linalg import inv

###########################################3
#2D car with a constant velocity (acceleration = 0)
#NEED TO ADJUST CODE FOR A 1X4 VECTOR STATE 
#CANT MEASURE VELOCITY
#SEE ALL THE NEW DIMENSIONS OF ALL THE MATRICES AND ADJUST IF NECCESERY!

dt=0.01
##########################################
#We have in our model of the system some kind of uncertainty
model_uncertainty_position_x = 1.0 #my selection: [m] model's position uncertainty  [sigma_px]
model_uncertainty_position_y = 1.0 #my selection: [m] model's position uncertainty[  [sigma_py]

model_uncertainty_velocity_x = 0.3 #my selection: [(m/s)] model's velocity uncertainty [sigma_vx]
model_uncertainty_velocity_y = 0.3 #my selection: [(m/s)] model's velocity uncertainty [sigma_vy]

#Hence the varience is for example 
#################################
#The sensor's variance can be for example from measurement noise...
sensor_uncertainty_position_x = 0.6 #my selection: [m] sensor's position uncertainty  [sigma_px]
sensor_uncertainty_position_y = 0.6 #my selection: [m] sensor's position uncertainty  [sigma_py]



##########################################
# Covariance matrix (Sigma)
# [
#     [sigma_px^2, sigma_px_py, sigma_px_vx, sigma_px_vy],  # Covariance between position x and other variables
#     [sigma_py_px, sigma_py^2, sigma_py_vx, sigma_py_vy],  # Covariance between position y and other variables
#     [sigma_vx_px, sigma_vx_py, sigma_vx^2, sigma_vx_vy],  # Covariance between velocity x and other variables
#     [sigma_vy_px, sigma_vy_py, sigma_vy_vx, sigma_vy^2]   # Covariance between velocity y and other variables
# ]

#the dynamic model is a 2D car - position and velocity - hence the covariance(the dependency) sigma_px_vx is linear: x=v*dt,
#same idea for sigma_py_vy. the rest is 0 for example: sigma_px_py = 0 and sigma_px_vy = 0.
#moreover the covariance matrix is symetrical hence : sigma_px_py = sigma_py_px.
#The P matrix (state covariance) represents the uncertainty of the state (position and velocity in this case), after incorporating the system dynamics and predictions.
P = np.matrix([[model_uncertainty_position_x**2 , 0 , model_uncertainty_velocity_x * dt , 0],
               [0, model_uncertainty_position_y**2 , 0 , model_uncertainty_velocity_y * dt ],
               [model_uncertainty_velocity_x * dt ,0,model_uncertainty_velocity_x**2, 0],
               [0 , model_uncertainty_velocity_y * dt , 0 ,model_uncertainty_velocity_y**2 ]])

###########################################################################
# 2D const velocity car model - Prediction matrix which can give us our next state.(position and velocity)
F = np.matrix([[1,0,dt,0],
               [0,1,0,dt],
               [0,0,1,0],
               [0,0,0,1]]) 

#Our sensors can mcan measure only POSx and POSy without VEL! [GPS for example]
H = np.matrix([[1,0,0,0],
               [0,1,0,0]])  
##################################################
# R matrix represents the measurement uncertainty and is directly determined by the properties of the sensors you're using. 
# It's typically independent of the dynamic model of the system
R = np.matrix([[sensor_uncertainty_position_x, 0],
               [0, sensor_uncertainty_position_y]])
#################################################
I = np.eye(4)


def filter(x, P, u, measurements):
    
    for n in range(len(measurements)):
        
        # prediction

        ###############Project the state ahead################
        x = F * x # from main we get x as the initial state! not measurement!///there is no controller-> u vector is 0 vector, Hence we neglect it.
        ###############Project the error covariance ahead################
        P = F * P * F.transpose() #there is no Q matrix cause we assume no additional uncceratinty from the enviroment
        
        # measurement update
        Z = np.matrix([[measurements[n][0]],
                        [measurements[n][1]]])  # Convert measurement to column vector
        y = H * x #mu(best estimate) of the predicted measurement converted into the measurement space!
        S =  H * P *H.transpose()#sigma (error) of the predicted measurement converted into the measurement space!
        K = P * H.transpose()*np.linalg.inv(S + R) #SEE THAT IT IS THE SAME! P * H.transpose()*np.linalg.inv(H * P * H.transpose()+ R)
        x =  x + K * (Z - y ) #SEE THAT IT IS THE SAME! x + K * (Z - H * x )
        P = (I - (K * H)) * P # this is simplification of P - K * H * P 
    
    print ('x= ',x)
    print('P= ',P)
    return P




def main():
    initial_xy = [2., 10.]
    x = np.array([[initial_xy[0]], [initial_xy[1]], [0.], [0.]]) # initial state (location and velocity in 2D [POSx,Posy,VELx,VELy] )
    measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]] #We can measure only POSx and POSy without VEL!
    u = np.array([[0.], [0.], [0.], [0.]]) 
    P2=filter(x, P, u, measurements)
    #print(P2[0][0])
   
    

if __name__ == '__main__':
    main()
