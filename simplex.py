import random
import time
import itertools

import gym
import numpy as np
import concurrent.futures

import pyglet
import yaml
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from controllers.mpc import MPC
from map.start_point import get_rand_start_point
from obstacle_map import draw
from controllers.drivers import GapFollower
import reachability.f110_reach as reach
from shapely.geometry import Point
from shapely.geometry import Polygon as shapely_poly



class GymRunner(object):

    def __init__(self, mpcs, gap_followers, map_obstacles, num):
        self.vertices_list = []
        self.gap_followers = gap_followers
        self.mpcs = mpcs
        self.control_count = 10
        self.intersect = False
        self.map_obstacles = map_obstacles
        self.ftg = False
        self.actions = None
        self.label = ""
        self.num_vehicles = num
        self.setup_env()
        self.setup_mpcs()
        self.setup_colors()

    def setup_env(self):
        with open('obstacle_map/new_config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)
        self.env = gym.make('f110_gym:f110-v0', map=self.conf.map_path, map_ext=self.conf.map_ext,
                            num_agents=self.num_vehicles)
        start_points = get_rand_start_point('map/Spielberg_raceline.csv', self.num_vehicles)
        env_array = []
        for start_point in start_points:
            point_array = start_point.split(";")
            env_array.append([float(point_array[1]), float(point_array[2]), float(point_array[3])])
        self.obs, self.step_reward, self.done, self.info = self.env.reset(np.array(env_array))
        self.env.render()

    def setup_mpcs(self):
        for i, mpc in enumerate(self.mpcs):
            mpc.setup(self.conf, self.env, i)

    def setup_colors(self):
        self.colors = []
        for i in range(self.num_vehicles):
            self.colors.append((random.randint(0, 250), random.randint(0, 250), random.randint(0, 250), 10))

    def mpc(self):
        actions = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for mpc in self.mpcs:
                futures.append(executor.submit(mpc.step, self.obs))

        all_car_polys = []
        for future, mpc, reach_color in itertools.zip_longest(futures, self.mpcs, self.colors):
            speed, steer, state = future.result()
            # run reachability and plot
            vertices_list, polys = reach.reachability(mpc.controller.oa, mpc.controller.odelta, mpc.num, state,
                                                               self.env.renderer.batch, reach_color)
            all_car_polys.append(polys)
            self.vertices_list += vertices_list
            actions.append([steer, speed])
        actions = np.array(actions)

        self.check_intersection(all_car_polys)

        return actions

    def follow_the_gap(self):
        MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
        MAX_SPEED = 5  # maximum speed [m/s]
        actions = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, driver in enumerate(self.gap_followers):
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
            self.vertices_list = []

    def check_zoom(self, zoom):
        if not zoom:
            self.env.renderer.bottom = 7 * -460 + 1500
            self.env.renderer.top = 7 * 460 + 1500
            self.env.renderer.left = 7 * -560.0
            self.env.renderer.right = 7 * 560.0

    def select_control(self, ftg_laptime):
        if self.control_count == 10:
            self.rm_plotted_reach_sets()
            self.actions = self.mpc()
            self.control_count = 0
            self.ftg = False
            self.label = "MPC"

        if self.intersect or self.ftg:
            self.actions = self.follow_the_gap()
            self.ftg = True
            ftg_laptime += self.step_reward
            self.label = "Follow the Gap"
            # time.sleep(0.1)

    def check_intersection(self, all_car_polys):
        for i, polys in enumerate(all_car_polys):
            for final_reach_poly in polys:
                reachpoly = shapely_poly(final_reach_poly.V)
                #check intersection with obstacles
                for map_obstacle in map_obstacles:
                    p = Point(map_obstacle)
                    c = p.buffer(0.75).boundary
                    if c.intersects(reachpoly):
                        self.intersect = True
                # check intersection of vehicles with one another
                for x in range(len(all_car_polys)-1-i):
                    for other_reachpoly in all_car_polys[-(x+1)]:
                        other_reachpoly = shapely_poly(other_reachpoly.V)
                        if other_reachpoly.intersects(reachpoly):
                            self.intersect = True


    def camera_follow(self, old_cam_point):
        # camera to follow vehicle
        camera_point = [self.obs['poses_x'][0] - old_cam_point[0], self.obs['poses_y'][0] - old_cam_point[1]]
        self.env.renderer.bottom += (camera_point[1] * 50)
        self.env.renderer.top += (camera_point[1] * 50)
        self.env.renderer.left += (camera_point[0] * 50)
        self.env.renderer.right += (camera_point[0] * 50)

    def end_sim(self, laptime, ftg_laptime, start):
        print("Lap completed!")
        print("follow the gap control %: ", (ftg_laptime / laptime) * 100)
        print("mpc control %: ", 100 - ((ftg_laptime / laptime) * 100))
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

    def run(self, zoom):
        # load map
        self.env.renderer.set_fullscreen(True)
        self.env.renderer.set_mouse_visible(False)
        start = time.time()
        laptime = 0
        ftg_laptime = 0

        pyglet_label = pyglet.text.Label('{label}'.format(
            label='MPC'), font_size=200, x=1000, y=1800, anchor_x='center', anchor_y='center',
            color=(255, 255, 255, 255), batch=self.env.renderer.batch)

        while not self.done and self.env.renderer.alive:
            self.intersect = False

            self.check_zoom(zoom)
            self.select_control(ftg_laptime)

            old_cam_point = [self.obs['poses_x'][0], self.obs['poses_y'][0]]
            pyglet_label.text = self.label
            self.obs, self.step_reward, self.done, self.info = self.env.step(self.actions)

            self.camera_follow(old_cam_point)
            laptime += self.step_reward
            self.env.render(mode='human_fast')

            if self.env.lap_counts[0] == 1:
                break;

            self.control_count += 1

        self.end_sim(laptime, ftg_laptime, start)


if __name__ == '__main__':
    parser = ArgumentParser(description="Settings",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-z", "--zoom", action="store_true", help="Zoom in camera")
    parser.add_argument("-n", "--number", type=int, default=1, help='number of vehicles')
    args = vars(parser.parse_args())
    zoom = args['zoom']
    num = args['number']
    map_obstacles = [[-58.51379908, 31.52080008], [-43.27495834, 37.9264539], [-48.63789174, 32.03631021],
                     [-30.77788556, 19.68154824], [-20.39962477, 24.76222363], [15.35970888, 25.54368615],
                     [22.28650099, 15.74832835], [17.20246417, 4.848844867]]
    draw.add_obstacles(map_obstacles)
    map_obstacles = [[-58.51379908, 31.52080008], [-43.27495834, 37.9264539], [-48.63789174, 32.03631021],
                     [-30.77788556, 19.68154824], [-20.39962477, 24.76222363], [15.35970888, 25.54368615],
                     [22.28650099, 15.74832835], [17.20246417, 4.848844867]]
    gap_followers = []
    mpcs = []
    for i in range(num):
        gap_followers.append(GapFollower())
        mpcs.append(MPC())
    runner = GymRunner(mpcs, gap_followers, map_obstacles, num)
    runner.run(zoom)
