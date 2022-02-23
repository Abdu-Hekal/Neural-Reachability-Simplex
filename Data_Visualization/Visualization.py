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
################################      Calculate Errors    ##############################################

# Calculate velocity error
velocity_error = np.array(data.control_velocity) -np.array(data.vehicle_velocity_x)
# Calculate heading error
heading_error = np.array(data.control_raceline_heading[1:]) -np.array(data.vehicle_position_heading[:-1])
# Filter Singularities in heading error because of F1TENTH Gym heading issue
heading_error = np.where(heading_error > 0.6, 0, heading_error)
heading_error = np.where(heading_error < -0.6, 0, heading_error)



# Calculate lateral error - deviation from the path
x_dist = np.array(data.vehicle_position_x[:-1]) -np.array(data.control_raceline_x[1:])  # Be careful: The logging of the raceline has one additional step
y_dist =np.array(data.vehicle_position_y[:-1]) -np.array(data.control_raceline_y[1:])   # Be careful: The logging of the raceline has one additional step
lateral_error = np.sqrt(pow(x_dist,2)+pow(y_dist,2))

###############################################################################################################
################################      Visualize Vehicle Data     ##############################################

# Creating the number of subplots
fig, axs = plt.subplots(3,2)

#  Velocity of the vehicle
axs[0,0].plot(data.vehicle_velocity_x, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
axs[0,0].plot(data.control_velocity, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Veloctiy')
axs[0,0].set_ylim([0, max(data.vehicle_velocity_x)+0.5])
axs[0,0].set_title('Vehicle Velocity: Actual Velocity vs. Raceline Velocity')
axs[0,0].set(ylabel='Velocity in m/s')
axs[0,0].grid(axis="both")
axs[0,0].legend()

#  Heading of the Vehicle
axs[1,0].plot(data.vehicle_position_heading , linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Heading')
axs[1,0].plot(data.control_raceline_heading, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Heading')
axs[1,0].set_title('Vehicle Heading: Actual Heading vs. Raceline Heading')
axs[1,0].set(ylabel='Vehicle Heading in rad')
axs[1,0].grid(axis="both")
axs[1,0].legend()


#  Steering Angle
axs[2,0].plot(data.steering_angle, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Heading')
axs[2,0].set_title('Steering angle')
axs[2,0].set(ylabel='Steering angle in degree')
axs[2,0].grid(axis="both")
axs[2,0].legend()


#########   ERRORS      #########

#  Velocity Error of the vehicle
axs[0,1].plot(velocity_error, linestyle ='solid',linewidth=2,color = '#005293', label = 'Veloctiy Error')
axs[0,1].set_ylim([-1.6, max(velocity_error)+0.2])
axs[0,1].set_title('Velocity Error')
axs[0,1].set(ylabel='Velocity in m/s')
axs[0,1].grid(axis="both")
axs[0,1].legend()

#  Heading Error of the vehicle
axs[1,1].plot(heading_error, linestyle ='solid',linewidth=2,color = '#005293', label = 'Heading Error')
axs[1,1].set_ylim([min(heading_error)-0.1, max(heading_error)+0.1])
axs[1,1].set_title('Heading Error')
axs[1,1].set(ylabel='Vehicle Heading in rad')
axs[1,1].grid(axis="both")
axs[1,1].legend()


#  Lateral Error of the vehicle
axs[2,1].plot(lateral_error, linestyle ='solid',linewidth=2,color = '#005293', label = 'Lateral Error')
#axs[2,1].plot(lateral_error_smoothed, linestyle ='solid',linewidth=2,color = '#00FF00', label = 'Actual Veloctiy')
axs[2,1].set_ylim([min(lateral_error)-0.2, max(lateral_error)+0.2])
axs[2,1].set_title('Lateral Error')
axs[2,1].set(ylabel='Distance in m')
axs[2,1].grid(axis="both")
axs[2,1].legend()

plt.show()

########## Vehicle Dynamics Plots
fig3, axs3 = plt.subplots(3)

#   Velocity of the vehicle    #######
axs3[0].plot(data.vehicle_velocity_x, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
axs3[0].plot(data.control_velocity, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Veloctiy')
axs3[0].set_ylim([0, max(data.vehicle_velocity_x)+0.5])
axs3[0].set_title('Vehicle Velocity: Actual Velocity vs. Raceline Velocity')
axs3[0].set(ylabel='Velocity in m/s')
axs3[0].grid(axis="both")

axs3[1].plot(long_accel, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
axs3[1].set_ylim([min(long_accel)-0.2, max(long_accel)+0.2])
axs3[1].set_title('Longitudinal Vehicle Acceleration')
axs3[1].set(ylabel='Acceleration in m/s2')
axs3[1].grid(axis="both")

#axs3[1].plot(long_accel, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
axs3[2].set_ylim([min(long_accel)-0.2, max(long_accel)+0.2])
axs3[2].set_title('Lateral Vehicle Acceleration')
axs3[2].set(ylabel='Acceleration in m/s2')
axs3[2].grid(axis="both")

plt.show()

###########################################    WHOLE TRACK      ############################################

#  Plot driven path of vehicle for all laps
plt.figure(1)
plt.plot(data.vehicle_position_x,data.vehicle_position_y,linestyle ='solid',linewidth=2, color = '#005293', label = 'Driven Path')
#plt.plot(raceline_x,raceline_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Path')
plt.plot(data.control_raceline_x,data.control_raceline_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Path')

plt.axis('equal')
plt.xlabel ('X-Position on track')
plt.ylabel ('Y-Position on track')
plt.legend()
plt.title ('Vehicle Path: Driven Path vs. Raceline Path')
plt.show()
