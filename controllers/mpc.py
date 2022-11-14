from controllers.mpc_tracking import LatticePlanner, Controller, State


class MPC:
    def __init__(self):
        self.path = None
        self.controller = None
        self.planner = None

    def setup(self, conf, env, car):
        self.planner = LatticePlanner(conf, env)
        self.controller = Controller(conf, car)
        # Load global raceline to create a path variable that includes all reference path information
        self.path = self.planner.plan()
        self.num = car.num

    def get_state(self, obs):
        state = State(x=obs['poses_x'][self.num], y=obs['poses_y'][self.num], yaw=obs['poses_theta'][self.num],
                      v=obs['linear_vels_x'][self.num])
        return state

    def get_control_action(self, obs):
        state = self.get_state(obs)
        speed, steer = self.planner.control(state.x, state.y, state.yaw,
                                            state.v, self.path, self.controller)

        return speed, steer, state

    def step(self, obs):
        return self.get_control_action(obs)
