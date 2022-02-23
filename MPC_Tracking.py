import cvxpy
import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
from numba import njit
import matplotlib.pyplot as plt
import pickle
import copy
import cubic_spline_planner

import pyglet
from pyglet.gl import *

import reach




#--------------------------- Controller Paramter ---------------------------
# System config
NX = 4          # state vector: z = [x, y, v, yaw]
NU = 2          # input vector: u = = [accel, steer]
T = 5           # finite time horizon length

# MPC parameters
R = np.diag([0.01, 100.0])              # input cost matrix, penalty for inputs - [accel, steer]
Rd = np.diag([0.01, 100.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]
Q = np.diag([0.01, 0.01, 0.5, 0.5])       # state cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
Qf = Q    # state final matrix, penalty  for the final state constraints: [x, y, v, yaw]

# Iterative paramter
MAX_ITER = 1                            # Max iteration
DU_TH = 0.01                             # Threshold for stopping iteration
N_IND_SEARCH = 5                        # Search index number
DT = 0.4 #0.10                               # time step [s]
dl = 1 #0.20                               # dist step [m]

# Vehicle parameters
LENGTH = 0.58                       # Length of the vehicle [m]
WIDTH = 0.31                        # Width of the vehicle [m]
WB = 0.33                           # Wheelbase [m]
MAX_STEER = np.deg2rad(24.0)        # maximum steering angle [rad]
#MAX_STEER = np.deg2rad(24.0)        # maximum steering angle [rad]         #REAL PARAMETER
MAX_DSTEER = np.deg2rad(180.0)       # maximum steering speed [rad/s]
#MAX_DSTEER = np.deg2rad(180.0)       # maximum steering speed [rad/s]      #REAL PARAMETER
MAX_SPEED = 6.5                    # maximum speed [m/s]
MIN_SPEED = 0                       # minimum backward speed [m/s]
MAX_ACCEL = 11.5                     # maximum acceleration [m/ss]

class CustomGroup(pyglet.graphics.Group):
    def set_state(self):
        glEnable(GL_TEXTURE_2D)

    def unset_state(self):
        glDisable(GL_TEXTURE_2D)

""" 
Planner Helpers
"""
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


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


class Controller:

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.mpc_initialize = 0
        self.target_ind = 0
        self.odelta = None
        self.oa = None
        self.origin_switch = 1

    def load_waypoints(self, conf):
        # Loading the x and y waypoints in the "..._raceline.vsv" that include the path to follow
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)


    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        """
        calc index of the nearest ref trajector in N steps
        :param node: path information X-Position, Y-Position, current index.
        :return: nearest index,
        """

        if pind == len(cx)-1:
            dx = [state.x - icx for icx in cx[0:(0 + N_IND_SEARCH)]]
            dy = [state.y - icy for icy in cy[0:(0 + N_IND_SEARCH)]]

            d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
            mind = min(d)
            ind = d.index(mind) + 0

        else:
            dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
            dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

            d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
            mind = min(d)
            ind = d.index(mind) + pind

        mind = math.sqrt(mind)
        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, dl, pind):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        ind, _ = Controller.calc_nearest_index(self, state, cx, cy, cyaw, pind)

        #if pind >= ind:
        #    ind = pind

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0                # steer operational point should be 0

        # Initialize Parameter
        travel = 0.0
        self.origin_switch = 1

        for i in range(T + 1):
            travel += abs(state.v) * DT     # Travel Distance into the future based on current velocity: s= v * t
            dind = int(round(travel / dl))  # Number of distance steps we need to look into the future

            if (ind + dind) < ncourse:
                ref_traj[0, i] = cx[ind + dind]
                ref_traj[1, i] = cy[ind + dind]
                ref_traj[2, i] = sp[ind + dind]

                # IMPORTANT: Take Care of Heading Change from 2pi -> 0 and 0 -> 2pi, so that all headings are the same
                if cyaw[ind + dind] -state.yaw > 5:
                    ref_traj[3, i] = abs(cyaw[ind + dind] -2* math.pi)
                elif cyaw[ind + dind] -state.yaw < -5:
                    ref_traj[3, i] = abs(cyaw[ind + dind] + 2 * math.pi)
                else:
                    ref_traj[3, i] = cyaw[ind + dind]

            else:
                # This function takes care about the switch at the origin/ Lap switch
                ref_traj[0, i] = cx[self.origin_switch]
                ref_traj[1, i] = cy[self.origin_switch]
                ref_traj[2, i] = sp[self.origin_switch]
                ref_traj[3, i] = cyaw[self.origin_switch]
                dref[0, i] = 0.0
                self.origin_switch = self.origin_switch +1

        return ref_traj, ind, dref

    def predict_motion(x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state = Controller.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(state, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * DT
        state.y = state.y + state.v * math.sin(state.yaw) * DT
        state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
        state.v = state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED
        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED

        return state

    @njit(fastmath=False, cache=True)
    def get_linear_model_matrix(v, phi, delta):
        """
           calc linear and discrete time dynamic model.
           :param v: speed: v_bar
           :param phi: angle of vehicle: phi_bar
           :param delta: steering angle: delta_bar
           :return: A, B, C
           """

        A = np.zeros((NX, NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = DT * math.cos(phi)
        A[0, 3] = - DT * v * math.sin(phi)
        A[1, 2] = DT * math.sin(phi)
        A[1, 3] = DT * v * math.cos(phi)
        A[3, 2] = DT * math.tan(delta) / WB

        B = np.zeros((NX, NU))
        B[2, 0] = DT
        B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

        C = np.zeros(NX)
        C[0] = DT * v * math.sin(phi) * phi
        C[1] = - DT * v * math.cos(phi) * phi
        C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

        return A, B, C

    def get_nparray_from_matrix(x):
        return np.array(x).flatten()

    def iterative_linear_mpc_control(ref_path, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if oa is None or od is None:
            oa = [0.0] * T
            od = [0.0] * T

        # Run the MPC calculation iterativly
        for i in range(MAX_ITER):

            # Call the Motion Prediction function. Prediction the vehicle motion for x-steps
            path_predict = Controller.predict_motion(x0, oa, od, ref_path)
            poa, pod = oa[:], od[:]

            # Call the Linear MPC Function
            mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = Controller.linear_mpc_control(ref_path, path_predict, x0, dref)

            # Calculta the u change value
            du = sum(abs(mpc_a - poa)) + sum(abs(mpc_delta - pod))
            if du <= DU_TH:
                break

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v

    def linear_mpc_control(ref_traj, path_predict, x0, dref):
        """
        solve the quadratic optimization problem using cvxpy, solver: OSQP

        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        dref: reference steer angle
        :return: optimal acceleration and steering strateg
        """

        # Initialize vectors
        x = cvxpy.Variable((NX, T + 1))     # Vehicle State Vector
        u = cvxpy.Variable((NU, T))         # Control Input vector
        cost = 0.0                          # Set costs to zero
        constraints = []                    # Create constraints array

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)

            if t != 0:
                cost += cvxpy.quad_form(ref_traj[:, t] - x[:, t], Q)

            A, B, C = Controller.get_linear_model_matrix(path_predict[2, t], path_predict[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

        cost += cvxpy.quad_form(ref_traj[:, T] - x[:, T], Qf)

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]


        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        #prob.solve(solver=cvxpy.GUROBI, verbose=True, warm_start= True)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)


        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = Controller.get_nparray_from_matrix(x.value[0, :])
            oy = Controller.get_nparray_from_matrix(x.value[1, :])
            ov = Controller.get_nparray_from_matrix(x.value[2, :])
            oyaw = Controller.get_nparray_from_matrix(x.value[3, :])
            oa = Controller.get_nparray_from_matrix(u.value[0, :])
            odelta = Controller.get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def MPC_Controller (self, vehicle_state, path):

        # --------------------------- Inititalize ---------------------------
        # Initialize the MPC parameter
        if self.mpc_initialize == 0:
            #self.target_ind, _ = Controller.calc_nearest_index(vehicle_state, cx, cy, cyaw, 0)
            self.target_ind = 0
            self.odelta, self.oa = None, None
            self.mpc_initialize = 1

        #------------------- MPC CONTROL LOOP ---------------------------------
        # Extract information about the path that needs to be followed
        cx = path[0]
        cy = path[1]
        cyaw = path[2]
        sp = path[4]

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        ref_path, self.target_ind, ref_delta = Controller.calc_ref_trajectory(self, vehicle_state, cx, cy, cyaw, sp, dl, self.target_ind)

        # Create State Vector based on current vehicle state: x-position, y-position,  velocity, heading
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the Linear MPC Control problem
        self.oa, self.odelta, ox, oy, oyaw, ov = Controller.iterative_linear_mpc_control(ref_path, x0, ref_delta, self.oa, self.odelta)
        print("oa: ", self.oa)
        print("ov: ", ov)
        print("ox: ", ox)

        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]

        ###########################################
        #                    DEBUG
        ##########################################

        debugplot = 0
        if debugplot == 1:
            plt.cla()
            # plt.axis([-40, 2, -10, 10])
            plt.axis([vehicle_state.x - 6, vehicle_state.x + 4.5, vehicle_state.y - 2.5, vehicle_state.y  + 2.5])
            plt.plot(self.waypoints[:, [1]], self.waypoints[:, [2]], linestyle='solid', linewidth=2, color='#005293', label='Raceline')
            plt.plot(vehicle_state.x, vehicle_state.y, marker='o', markersize=10, color='red', label='CoG')
            plt.plot(ref_path[0], ref_path[1], linestyle='dotted', linewidth=8, color='purple',label='MPC Input: Ref. Trajectory for T steps')
            #plt.plot(cx[self.target_ind], cy[self.target_ind], marker='x', markersize=10, color='green',)
            plt.plot(ox, oy, linestyle='dotted', linewidth=5, color='green',label='MPC Output: Trajectory for T steps')
            plt.legend()
            plt.pause(0.001)
            plt.axis('equal')

        debugplot2 = 0
        if debugplot2 == 1:
            plt.cla()
            # Creating the number of subplots
            fig, axs = plt.subplots(3, 1)
            #  Velocity of the vehicle
            axs[0].plot(ov, linestyle='solid', linewidth=2, color='#005293')
            axs[0].set_ylim([0, max(ov) + 0.5])
            axs[0].set(ylabel='Velocity in m/s')
            axs[0].grid(axis="both")

            axs[1].plot(self.oa, linestyle='solid', linewidth=2, color='#005293')
            axs[1].set_ylim([0, max(self.oa) + 0.5])
            axs[1].set(ylabel='Acceleration in m/s')
            axs[1].grid(axis="both")
            plt.pause(0.001)
            plt.axis('equal')

        ###########################################
        #                    DEBUG
        ##########################################


        #------------------- MPC CONTROL Output ---------------------------------
        # Create the final steer and speed parameter that need to be sent out

        # Steering Output: First entry of the MPC steering angle output vector in degree
        steer_output = self.odelta[0]

        # Acceleration Output: First entry of the MPC acceleration output in m/s2
        # The F1TENTH Gym needs velocity as an control input: Acceleration -> Velocity
        # accelerate
        #speed_output=self.oa[0]*T*DT
        #speed_output=ref_path[2][1]*0.50
        speed_output= vehicle_state.v + self.oa[0] * DT

        #print ("Current Speed:", vehicle_state.v, "Control Speed:", speed,"MPC Speed:",ov)
        #print("Vehicle Heading: ", vehicle_state.yaw, "MPC Heading:",oyaw[0], "RefPath Heading:",ref_path[3], "Racline Heading:",cyaw[self.target_ind])
        #print ("Vehicle X:", vehicle_state.x, "Target X:",cx[self.target_ind],"----- Vehicle Y:", vehicle_state.y, " Target Y:",cy[self.target_ind])

        return speed_output, steer_output



class LatticePlanner:

    def __init__(self, conf, env, wb):
        self.conf = conf                        # Current configuration for the gym based on the maps
        self.env = env                          # Current environment parameter
        self.load_waypoints(conf)               # Waypoints of the raceline
        self.init_flag = 0                      # Initialization of the states
        self.calcspline = 0                     # Flag for Calculation the Cubic Spline
        self.initial_state = []

    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def plan(self):
        """
        Loading the individual data from the global, optimal raceline and creating one list
        """

        cx = self.waypoints[:, 1]       # X-Position of Raceline
        cy = self.waypoints[:, 2]       # Y-Position of Raceline
        cyaw = self.waypoints[:, 3]     # Heading on Raceline
        ck = self.waypoints[:, 4]       # Curvature of Raceline
        cv = self.waypoints[:, 5]       # velocity on Raceline

        global_raceline = [cx, cy, cyaw, ck, cv]

        return global_raceline

    def control(self, pose_x, pose_y, pose_theta, velocity, path):
        """
        Control loop for calling the controller
        """

        # -------------------- INITIALIZE Controller ----------------------------------------
        if self.init_flag == 0:
            vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=0.1)
            self.init_flag = 1
        else:
            vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=velocity)

        # -------------------- Call the MPC Controller ----------------------------------------
        speed, steering_angle = controller.MPC_Controller(vehicle_state, path)

        return speed, steering_angle


# -------------------------- MAIN SIMULATION  ----------------------------------------

if __name__ == '__main__':
    # Check CVXP Installations
    #print(cvxpy.installed_solvers())

    # Load the configuration for the desired Racetrack
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.80}
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Create the simulation environment and inititalize it
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Creating the Motion planner and Controller object that is used in Gym
    planner = LatticePlanner(conf, env, 0.17145 + 0.15875)
    controller = Controller(conf, 0.17145 + 0.15875)

    # Creating a Datalogger object that saves all necessary vehicle data
    logging = Datalogger(conf)

    # Initialize Simulation
    laptime = 0.0
    control_count = 10
    start = time.time()

    # Load global raceline to create a path variable that includes all reference path information
    path = planner.plan()

    window = env.renderer #this is the EnvRenderer which inherits from pyglet.window.Window.
    # setting "batch=window.batch" tells it to plot on the f1tenth renderer window
    

    # -------------------------- SIMULATION LOOP  ----------------------------------------
    while not done:

        # initialise vertix list
        vertex_list = None
        #AH: add a vertices to batch and plot them
        # Call the function for planning a path, only every 10th timestep
        if control_count == 10:

            # Call the0 function for tracking speed and steering
            # MPC specific: We solve the MPC problem only every 10th timestep of the simultation to decrease the sim time
            print("observed x: ", obs['poses_x'][0])
            print("observed y: ", obs['poses_y'][0])
            #create a state class where we instantiate the vehicle's current state to pass to reachability analysis
            state = State(x=obs['poses_x'][0], y=obs['poses_y'][0], yaw=obs['poses_theta'][0], v=obs['linear_vels_x'][0])
            #run reachability and plot
            vertex_list = window.batch.add(4, pyglet.gl.GL_POLYGON, CustomGroup(), ('v2i', (100, 100,150, 100,150, 150,100, 150)))

            speed, steer = planner.control(state.x, state.y, state.yaw, state.v, path)
            control_count = 0

        # Update the simulation environment

        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        env.render(mode='human_fast')

        #AH: removes added vertex
        if vertex_list:
            vertex_list.delete()

        # Apply Looging to log information from the waypoints

        if conf_dict['logging'] == 'True':
            logging.logging(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0],
                            obs['linear_vels_y'][0], obs['lap_counts'], speed, steer)

            wpts = np.vstack((planner.waypoints[:, conf.wpt_xind], planner.waypoints[:, conf.wpt_yind])).T
            vehicle_state = np.array([obs['poses_x'][0], obs['poses_y'][0]])
            nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(vehicle_state, wpts)
            logging.logging2(nearest_point[0], nearest_point[1], path[2][i])

        # Update Asynchronous Counter for the MPC loop
        control_count = control_count + 1
    if conf_dict['logging'] == 'True':
        pickle.dump(logging, open("Data_Visualization/datalogging_MPC.p", "wb"))

    # Print Statement that simulation is over
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
