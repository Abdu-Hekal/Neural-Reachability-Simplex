import random
from controllers.new_mpc_tracking import LatticePlanner, Controller, State


class MPC:
    def __init__(self):
        self.path = None
        self.controller = None
        self.planner = None

    def setup(self, conf, env, num):
        self.planner = LatticePlanner(conf, env)
        self.controller = Controller(conf)
        # Load global raceline to create a path variable that includes all reference path information
        self.path = self.planner.plan()
        self.num = num

    def step(self, obs):
        state = State(x=obs['poses_x'][self.num], y=obs['poses_y'][self.num], yaw=obs['poses_theta'][self.num],
                      v=obs['linear_vels_x'][self.num])
        speed, steer = self.planner.control(obs['poses_x'][self.num], obs['poses_y'][self.num], obs['poses_theta'][self.num],
                                            obs['linear_vels_x'][self.num], self.path, self.controller)

        return speed, steer, state
