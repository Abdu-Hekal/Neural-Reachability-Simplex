from controllers.mpc import MPC


class ReachsetMPC(MPC):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        pass
