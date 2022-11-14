from controllers.mpc import MPC
import reachability.f110_reach as reach


class ReachsetMPC(MPC):
    # assume for now that our time tc=0.1, will later change this
    def __init__(self, comp_time=0.1):
        super().__init__()
        self.comp_time = comp_time

    def step(self, obs):
        prev_oa, prev_odelta = self.controller.oa, self.controller.odelta
        state = self.get_state(obs)  # "state" is the state before the step
        if prev_oa is not None:
            last_poly = reach.reachability(state, prev_oa, prev_odelta,
                                           time_horizon_min=self.comp_time - 0.01, time_horizon_max=self.comp_time,
                                           plot=False)[1][-1]
            print("current state: ", state.x, state.y)
            print("predicted poly centroid: ", last_poly.centroid)
            vel_min_list, vel_max_list = reach.compute_velocity(prev_oa, prev_odelta, state,
                                                                time_horizon_min=self.comp_time - 0.01,
                                                                time_horizon_max=self.comp_time)
            median_vel = (vel_min_list[0][0][0] + vel_max_list[0][0][0])/2
            print("current velocity: ", state.v)
            print("predicted median velocity: ", median_vel)

        return self.get_control_action(obs)
