import time
import gym
import numpy as np
import concurrent.futures

import pyglet
import yaml
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

import draw
import reach
from controllers.MPC_Tracking import LatticePlanner, Controller, State
from drivers import GapFollower

# choose your drivers here (1-4)
drivers = [GapFollower()]


class GymRunner(object):

    def __init__(self, drivers, map_obstacles):
        self.vertices_list = None
        self.drivers = drivers
        self.setup_env()
        self.setup_mpc()
        self.laptime = 0.0
        self.control_count = 10
        self.intersect = False
        self.map_obstacles = map_obstacles

    def setup_env(self):
        with open('obstacle_map/new_config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)

        self.env = gym.make('f110_gym:f110-v0', map=self.conf.map_path, map_ext=self.conf.map_ext, num_agents=1)
        self.obs, self.step_reward, self.done, self.info = self.env.reset(
            np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]
                      ]))
        self.env.render()

    def setup_mpc(self):
        self.planner = LatticePlanner(self.conf, self.env)
        self.controller = Controller(self.conf)
        # Load global raceline to create a path variable that includes all reference path information
        self.path = self.planner.plan()

    def mpc(self):
        state = State(x=self.obs['poses_x'][0], y=self.obs['poses_y'][0], yaw=self.obs['poses_theta'][0],
                      v=self.obs['linear_vels_x'][0])
        speed, steer = self.planner.control(self.obs['poses_x'][0], self.obs['poses_y'][0], self.obs['poses_theta'][0],
                                            self.obs['linear_vels_x'][0], self.path, self.controller)

        # run reachability and plot
        self.vertices_list, self.intersect = reach.reachability(self.controller.oa, self.controller.odelta, state,
                                                                self.env.renderer.batch, self.map_obstacles)

        actions = np.array([[steer, speed]])

        return actions

    def follow_the_gap(self):
        MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
        MAX_SPEED = 5  # maximum speed [m/s]
        actions = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, driver in enumerate(drivers):
                futures.append(executor.submit(driver.process_lidar, self.obs['scans'][i]))
        for future in futures:
            speed, steer = future.result()
            speed = min(speed, MAX_SPEED)
            steer = min(steer, MAX_STEER)
            actions.append([steer, speed])
        actions = np.array(actions)

        return actions

    def rm_plotted_reach_sets(self):
        # AH: removes added vertex
        if self.vertices_list:
            for vertex_list in self.vertices_list:
                vertex_list.delete()
            self.vertices_list = None

    def run(self, zoom):
        # load map
        self.env.renderer.set_fullscreen(True)
        self.env.renderer.set_mouse_visible(False)
        start = time.time()
        ftg_laptime = 0

        pyglet_label = pyglet.text.Label('{label}'.format(
            label='MPC'), font_size=200, x=1000, y=1800, anchor_x='center', anchor_y='center',
            color=(255, 255, 255, 255), batch=self.env.renderer.batch)

        while not self.done and self.env.renderer.alive:

            if not zoom:
                self.env.renderer.bottom = 7 * -460 +1500
                self.env.renderer.top = 7 * 460 +1500
                self.env.renderer.left = 7 * -560.0
                self.env.renderer.right = 7 * 560.0

            if self.control_count == 10:
                self.rm_plotted_reach_sets()
                actions = self.mpc()
                self.control_count = 0
                ftg = False
                label = "MPC"

            if self.intersect or ftg:
                actions = self.follow_the_gap()
                ftg = True
                ftg_laptime += self.step_reward
                label = "Follow the Gap"
                # time.sleep(0.1)

            old_cam_point = [self.obs['poses_x'][0], self.obs['poses_y'][0]]

            pyglet_label.text = label
            self.obs, self.step_reward, self.done, self.info = self.env.step(actions)

            # camera to follow vehicle
            camera_point = [self.obs['poses_x'][0] - old_cam_point[0], self.obs['poses_y'][0] - old_cam_point[1]]
            self.env.renderer.bottom += (camera_point[1] * 50)
            self.env.renderer.top += (camera_point[1] * 50)
            self.env.renderer.left += (camera_point[0] * 50)
            self.env.renderer.right += (camera_point[0] * 50)

            self.laptime += self.step_reward

            self.env.render(mode='human_fast')

            if self.env.lap_counts[0] == 1:
                print("Lap completed!")
                print("follow the gap control %: ", (ftg_laptime / self.laptime) * 100)
                print("mpc control %: ", 100 - ((ftg_laptime / self.laptime) * 100))
                break;

            self.control_count += 1

        print('Sim elapsed time:', self.laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    parser = ArgumentParser(description="Settings",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-z", "--zoom", action="store_true", help="Zoom in camera")
    args = vars(parser.parse_args())
    zoom = args['zoom']
    map_obstacles = [[-58.51379908, 31.52080008], [-43.27495834, 37.9264539], [-48.63789174, 32.03631021],
                     [-30.77788556, 19.68154824], [-20.39962477, 24.76222363], [15.35970888, 25.54368615],
                     [22.28650099, 15.74832835], [17.20246417, 4.848844867]]
    draw.add_obstacles(map_obstacles)
    map_obstacles = [[-58.51379908, 31.52080008], [-43.27495834, 37.9264539], [-48.63789174, 32.03631021],
                     [-30.77788556, 19.68154824], [-20.39962477, 24.76222363], [15.35970888, 25.54368615],
                     [22.28650099, 15.74832835], [17.20246417, 4.848844867]]
    runner = GymRunner(drivers, map_obstacles)
    runner.run(zoom)
