from controllers.mpc import MPC
import reachability.f110_reach as reach


class ReachsetMPC(MPC):
    def __init__(self, comp_time=0.1):
        super().__init__()
        self.comp_time = comp_time
        self.iter = 1

    def step(self, obs):
        prev_oa, prev_odelta = self.controller.oa, self.controller.odelta
        state = self.get_state(obs)  # "state" is the state before the step
        if prev_oa is not None:
            print(prev_oa[self.iter:])
            last_poly = reach.reachability(state, prev_oa[self.iter:], prev_odelta[self.iter:],
                                           time_horizon=self.comp_time, plot=False)[1][-1]
            print("state: ", state.x, state.y)
            print("poly centroid: ", last_poly.centroid)

        return self.get_control_action(obs)
