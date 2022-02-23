import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as pltcol

class Datalogger:
    """
    This is the class for logging vehicle data in the F1TENTH Gym
    """
    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def __init__(self, conf):
        self.conf = conf                    # Current configuration for the gym based on the maps
        self.load_waypoints(conf)           # Waypoints of the raceline
        self.vehicle_position_x = []        # Current vehicle position X (rear axle) on the map
        self.vehicle_position_y = []        # Current vehicle position Y (rear axle) on the map
        self.vehicle_position_heading = []  # Current vehicle heading on the map
        self.vehicle_velocity_x = []        # Current vehicle velocity - Longitudinal
        self.vehicle_velocity_y = []        # Current vehicle velocity - Lateral
        self.control_velocity = []          # Desired vehicle velocity based on control calculation
        self.steering_angle = []            # Steering angle based on control calculation
        self.lapcounter = []                # Current Lap
        self.control_raceline_x = []        # Current Control Path X-Position on Raceline
        self.control_raceline_y = []        # Current Control Path y-Position on Raceline
        self.control_raceline_heading = []  # Current Control Path Heading on Raceline


    def logging(self, pose_x, pose_y, pose_theta, current_velocity_x, current_velocity_y, lap, control_veloctiy, control_steering):
        self.vehicle_position_x.append(pose_x)
        self.vehicle_position_y.append(pose_y)
        self.vehicle_position_heading.append(pose_theta)
        self.vehicle_velocity_x .append(current_velocity_x)
        self.vehicle_velocity_y.append(current_velocity_y)
        self.control_velocity.append(control_veloctiy)
        self.steering_angle.append(control_steering)
        self.lapcounter.append(lap)

    def logging2(self, raceline_x, raceline_y, raceline_theta):
        self.control_raceline_x.append(raceline_x)
        self.control_raceline_y.append(raceline_y)
        self.control_raceline_heading.append(raceline_theta)


# Load the pickle file data
current_path = os.getcwd()

# Load the correct filename
# PurePursuit,Stanley, LQR, MPC
filename = current_path + '/datalogging_PurePursuit.p'
file_to_read = open(filename, "rb")
data = pickle.load(file_to_read)

# Extract Raceline data
raceline_x = data.waypoints[:,[1]]
raceline_y = data.waypoints[:,[2]]
raceline_heading = data.waypoints[:,[3]]



########### Calculate additional vehicle parameters

y = 0
long_accel = []
long_velocity=np.array(data.vehicle_velocity_x)
for x in range(1,len(long_velocity)):
    v2 =long_velocity[x]
    v1 =long_velocity[y]
    delta_v = v2-v1
    accel =  delta_v  /0.01
    if accel < -50:
        accel =0
    long_accel.append(accel)
    y = y+1


###############################################################################################################
################################      Calculate Errors 1    ##############################################

# Calculate velocity error
velocity_error1 = np.array(data.control_velocity) -np.array(data.vehicle_velocity_x)
# Calculate heading error
heading_error1 = np.array(data.control_raceline_heading[1:]) -np.array(data.vehicle_position_heading[:-1])
heading_error1 = np.where(heading_error1 > 0.6, 0, heading_error1)
heading_error1 = np.where(heading_error1 < -0.6, 0, heading_error1)

# Calculate lateral error - deviation from the path
x_dist = np.array(data.vehicle_position_x[:-1]) -np.array(data.control_raceline_x[1:])  # Be careful: The logging of the raceline has one additional step
y_dist =np.array(data.vehicle_position_y[:-1]) -np.array(data.control_raceline_y[1:])   # Be careful: The logging of the raceline has one additional step
lateral_error1 = np.sqrt(pow(x_dist,2)+pow(y_dist,2))



##############################################################################################################################################################################################################################
###############################################################################################################
###############################################################################################################
################################      Load Data  2   ##############################################

# Load the correct filename
# PurePursuit,Stanley, LQR, MPC
filename = current_path + '/datalogging_Stanley.p'
file_to_read = open(filename, "rb")
data = pickle.load(file_to_read)

# Extract Raceline data
raceline_x = data.waypoints[:,[1]]
raceline_y = data.waypoints[:,[2]]
raceline_heading = data.waypoints[:,[3]]

########### Calculate additional vehicle parameters

y = 0
long_accel = []
long_velocity=np.array(data.vehicle_velocity_x)
for x in range(1,len(long_velocity)):
    v2 =long_velocity[x]
    v1 =long_velocity[y]
    delta_v = v2-v1
    accel =  delta_v  /0.01
    if accel < -50:
        accel =0
    long_accel.append(accel)
    y = y+1

###############################################################################################################
################################      Calculate Errors 2    ##############################################

# Calculate velocity error
velocity_error2 = np.array(data.control_velocity) -np.array(data.vehicle_velocity_x)
# Calculate heading error
heading_error2 = np.array(data.control_raceline_heading[1:]) -np.array(data.vehicle_position_heading[:-1])
heading_error2 = np.where(heading_error2 > 0.6, 0, heading_error2)
heading_error2 = np.where(heading_error2 < -0.6, 0, heading_error2)

# Calculate lateral error - deviation from the path
x_dist2 = np.array(data.vehicle_position_x[:-1]) -np.array(data.control_raceline_x[1:])  # Be careful: The logging of the raceline has one additional step
y_dist2 =np.array(data.vehicle_position_y[:-1]) -np.array(data.control_raceline_y[1:])   # Be careful: The logging of the raceline has one additional step
lateral_error2 = np.sqrt(pow(x_dist2,2)+pow(y_dist2,2))


##############################################################################################################################################################################################################################
###############################################################################################################
###############################################################################################################
################################      Load Data 3    ##############################################

# Load the correct filename
# PurePursuit,Stanley, LQR, MPC
filename = current_path + '/datalogging_LQR.p'
file_to_read = open(filename, "rb")
data = pickle.load(file_to_read)

# Extract Raceline data
raceline_x = data.waypoints[:,[1]]
raceline_y = data.waypoints[:,[2]]
raceline_heading = data.waypoints[:,[3]]

########### Calculate additional vehicle parameters

y = 0
long_accel = []
long_velocity=np.array(data.vehicle_velocity_x)
for x in range(1,len(long_velocity)):
    v2 =long_velocity[x]
    v1 =long_velocity[y]
    delta_v = v2-v1
    accel =  delta_v  /0.01
    if accel < -50:
        accel =0
    long_accel.append(accel)
    y = y+1

###############################################################################################################
################################      Calculate Errors 3    ##############################################

# Calculate velocity error
velocity_error3 = np.array(data.control_velocity) -np.array(data.vehicle_velocity_x)
# Calculate heading error
heading_error3 = np.array(data.control_raceline_heading[1:]) -np.array(data.vehicle_position_heading[:-1])
heading_error3 = np.where(heading_error3 > 0.6, 0, heading_error3)
heading_error3 = np.where(heading_error3 < -0.6, 0, heading_error3)

# Calculate lateral error - deviation from the path
x_dist3 = np.array(data.vehicle_position_x[:-1]) -np.array(data.control_raceline_x[1:])  # Be careful: The logging of the raceline has one additional step
y_dist3 =np.array(data.vehicle_position_y[:-1]) -np.array(data.control_raceline_y[1:])   # Be careful: The logging of the raceline has one additional step
lateral_error3 = np.sqrt(pow(x_dist3,2)+pow(y_dist3,2))



##############################################################################################################################################################################################################################
###############################################################################################################
###############################################################################################################
################################      Load Data 4    ##############################################

# Load the correct filename
# PurePursuit,Stanley, LQR, MPC
filename = current_path + '/datalogging_MPC.p'
file_to_read = open(filename, "rb")
data = pickle.load(file_to_read)

# Extract Raceline data
raceline_x = data.waypoints[:,[1]]
raceline_y = data.waypoints[:,[2]]
raceline_heading = data.waypoints[:,[3]]

########### Calculate additional vehicle parameters

y = 0
long_accel = []
long_velocity=np.array(data.vehicle_velocity_x)
for x in range(1,len(long_velocity)):
    v2 =long_velocity[x]
    v1 =long_velocity[y]
    delta_v = v2-v1
    accel =  delta_v  /0.01
    if accel < -50:
        accel =0
    long_accel.append(accel)
    y = y+1

###############################################################################################################
################################      Calculate Errors 4    ##############################################

# Calculate velocity error
velocity_error4 = np.array(data.control_velocity) -np.array(data.vehicle_velocity_x)
# Calculate heading error
heading_error4 = np.array(data.control_raceline_heading[1:]) -np.array(data.vehicle_position_heading[:-1])
heading_error4 = np.where(heading_error4 > 0.6, 0, heading_error4)
heading_error4 = np.where(heading_error4 < -0.6, 0, heading_error4)

# Calculate lateral error - deviation from the path
x_dist4 = np.array(data.vehicle_position_x) -np.array(data.control_raceline_x)  # Be careful: The logging of the raceline has one additional step
y_dist4 =np.array(data.vehicle_position_y) -np.array(data.control_raceline_y)   # Be careful: The logging of the raceline has one additional step
lateral_error4 = np.sqrt(pow(x_dist4,2)+pow(y_dist4,2))


###############################################################################################################
################################      Visualize Vehicle Data     ##############################################

# Creating the number of subplots
fig, axs = plt.subplots(3,1)

#########   ERRORS      #########

#  Velocity Error of the vehicle
axs[0].plot(velocity_error1, linestyle ='solid',linewidth=2,color = '#005293', label = 'Pure Pursuit')
axs[0].plot(velocity_error2, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Stanley')
axs[0].plot(velocity_error3, linestyle ='dashed',linewidth=2, color = '#a2ad00', label = 'LQR')
axs[0].plot(velocity_error4, linestyle ='dashed',linewidth=2, color = '#64a0c8', label = 'MPC')
axs[0].set_ylim([-1.6, max(velocity_error1)+0.2])
axs[0].set_title('Velocity Error')
axs[0].set(ylabel='Velocity in m/s')
axs[0].grid(axis="both")
axs[0].legend()

#  Heading Error of the vehicle
axs[1].plot(heading_error1, linestyle ='solid',linewidth=2,color = '#005293', label = 'Pure Pursuit')
axs[1].plot(heading_error2, linestyle ='solid',linewidth=2, color = '#e37222', label = 'Stanley')
axs[1].plot(heading_error3, linestyle ='solid',linewidth=2, color = '#a2ad00', label = 'LQR')
axs[1].plot(heading_error4, linestyle ='solid',linewidth=2, color = '#64a0c8', label = 'MPC')
axs[1].set_ylim([min(heading_error1)-0.1, max(heading_error1)+0.1])
axs[1].set_title('Heading Error')
axs[1].set(ylabel='Vehicle Heading in rad')
axs[1].grid(axis="both")
axs[1].legend()

#  Lateral Error of the vehicle
axs[2].plot(lateral_error1, linestyle ='solid',linewidth=2,color = '#005293', label = 'Pure Pursuit')
axs[2].plot(lateral_error2, linestyle ='solid',linewidth=2, color = '#e37222', label = 'Stanley')
axs[2].plot(lateral_error3, linestyle ='solid',linewidth=2, color = '#a2ad00', label = 'LQR')
axs[2].plot(lateral_error4, linestyle ='solid',linewidth=2, color = '#64a0c8', label = 'MPC')
#axs[2,1].plot(lateral_error_smoothed, linestyle ='solid',linewidth=2,color = '#00FF00', label = 'Actual Veloctiy')
axs[2].set_ylim([min(lateral_error1)-0.2, max(lateral_error1)+0.2])
axs[2].set_title('Lateral Error')
axs[2].set(ylabel='Distance in m')
axs[2].grid(axis="both")
axs[2].legend()
plt.show()