class Car:
    def __init__(self, num=0, advanced_controller=None, baseline_controller=None, conf=None):
        self.num = num
        self.control_count = 10
        self.advanced_controller = advanced_controller
        self.baseline_controller = baseline_controller
        self.action = None
        self.intersect = False
        self.baseline = False
        self.reachset = []
        self.baseline_laptime = 0
        self.vertices_list = []
        self.label = ""
        self.conf = conf

        # Vehicle parameters
        self.LENGTH = 0.58  # Length of the vehicle [m]
        self.WIDTH = 0.31  # Width of the vehicle [m]
        self.WB = 0.33  # Wheelbase [m]
        self.MAX_STEER = 0.4189  # maximum steering angle [rad] from f1tenth gym library
        self.MAX_DSTEER = 3.2  # maximum steering speed [rad/s] from f1tenth gym library
        self.MAX_SPEED = 5  # maximum speed [m/s] from training data on flowstar
        self.MIN_SPEED = 0  # minimum backward speed [m/s]
        self.MAX_ACCEL = 9.51  # maximum acceleration [m/ss] from f1tenth gym library

    def __getstate__(self):
        out = self.__dict__.copy()
        del out["advanced_controller"]
        del out["baseline_controller"]
        del out["vertices_list"]
        return out
    def __setstate__(self, out):
        self.__dict__ = out

    def rm_plotted_reach_sets(self):
        # AH: removes added vertex
        if self.vertices_list:
            for vertex_list in self.vertices_list:
                vertex_list.delete()
            self.vertices_list = []