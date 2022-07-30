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
import multiprocessing as mp


class Car:
    def __init__(self, mpc_controller, ftg_controller):
        self.control_count = 10
        self.mpc_controller = mpc_controller
        self.ftg_controller = ftg_controller
        self.action = None
        self.future = None
        self.intersect = False
        self.ftg = False
        self.reachset = []
        self.ftg_laptime = 0
        self.vertices_list = []

    def __getstate__(self):
        return self.reachset
    def __setstate__(self, reachset):
        self.reachset = reachset

    def rm_plotted_reach_sets(self):
        # AH: removes added vertex
        if self.vertices_list:
            for vertex_list in self.vertices_list:
                vertex_list.delete()
            self.vertices_list = []



class GymRunner(object):

    def __init__(self, cars, map_obstacles):
        self.cars = cars
        self.map_obstacles = map_obstacles
        self.label = ""
        self.setup_env()
        self.setup_mpcs()
        self.setup_colors()
        self.actions = None

    def __getstate__(self):
        return self.cars
    def __setstate__(self, cars):
        self.cars = cars

    def setup_env(self):
        with open('obstacle_map/new_config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)
        self.env = gym.make('f110_gym:f110-v0', map=self.conf.map_path, map_ext=self.conf.map_ext,
                            num_agents=len(self.cars))
        start_points = get_rand_start_point('map/Spielberg_raceline.csv', len(self.cars))
        env_array = []
        for start_point in start_points:
            point_array = start_point.split(";")
            env_array.append([float(point_array[1]), float(point_array[2]), float(point_array[3])])
            #env_array.append([-0.0440806, -0.8491629, 3.4034118])
        self.obs, self.step_reward, self.done, self.info = self.env.reset(np.array(env_array))
        self.env.render()

    def setup_mpcs(self):
        for i, car in enumerate(cars):
            car.mpc_controller.setup(self.conf, self.env, i)

    def setup_colors(self):
        self.colors = []
        for i in range(len(self.cars)):
            self.colors.append((random.randint(0, 250), random.randint(0, 250), random.randint(0, 250), 10))

    def select_control(self, ftg_laptime):
        actions = []

        for index, (car,reach_color)  in enumerate(zip(self.cars,self.colors)):
            car.future = None
            self.check_intersection(car, index)
            if car.control_count == 10:
                car.rm_plotted_reach_sets()
                car.ftg = False
                self.label = "MPC"
                car.future = car.mpc_controller.step(self.obs)
                car.control_count = 0
            if car.intersect or car.ftg:
                car.ftg = True
                self.label = "Follow the Gap"
                car.ftg_laptime += self.step_reward
                car.future = car.ftg_controller.process_lidar(self.obs['scans'][index])

            if car.future:
                speed, steer, state = car.future
                car.action = [steer, speed]
                if not car.ftg:
                    vertices_list, polys = reach.reachability(car.mpc_controller.controller.oa, car.mpc_controller.controller.odelta,
                                                              car.mpc_controller.num, state, self.env.renderer.batch, reach_color)
                    car.vertices_list += vertices_list
                    car.reachset = polys

            if car.action:
                actions.append(car.action)
            else:
                raise Exception("No valid action given for car")
            car.control_count += 1

        self.actions = np.array(actions)

    def check_zoom(self, zoom):
        if not zoom:
            self.env.renderer.bottom = 7 * -460 + 1500
            self.env.renderer.top = 7 * 460 + 1500
            self.env.renderer.left = 7 * -560.0
            self.env.renderer.right = 7 * 560.0

    def check_intersection(self, car, index):
        car.intersect = False

        result_objects = [pool.apply_async(self.check_one_poly_intersection, (final_reach_poly, index)) for final_reach_poly in car.reachset]
        # x = [2]*10
        # result_objects = [pool.apply_async(my_try, (z, x)) for z in range(10)]
        intersect = [r.get() for r in result_objects]
        print(any(intersect))


    def check_one_poly_intersection(self, final_reach_poly, index):
        intersect = False
        reachpoly = shapely_poly(final_reach_poly.V)
        map_obstacles = [[-58.51379908, 31.52080008], [-43.27495834, 37.9264539], [-48.63789174, 32.03631021],
                         [-30.77788556, 19.68154824], [-20.39962477, 24.76222363], [15.35970888, 25.54368615],
                         [22.28650099, 15.74832835], [17.20246417, 4.848844867]]
        # check intersection with obstacles
        for map_obstacle in map_obstacles:
            p = Point(map_obstacle)
            c = p.buffer(0.75).boundary
            if c.intersects(reachpoly):
                intersect = True
        # check intersection of vehicles with one another
        for x in range(len(self.cars) - 1 - index):
            other_car = self.cars[-(x + 1)]
            for other_reachpoly in other_car.reachset:
                other_reachpoly = shapely_poly(other_reachpoly.V)
                if other_reachpoly.intersects(reachpoly):
                    intersect = True
                    other_car.intersect = True

        return intersect


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

            self.check_zoom(zoom)
            self.select_control(ftg_laptime)

            old_cam_point = [self.obs['poses_x'][0], self.obs['poses_y'][0]]
            pyglet_label.text = self.label
            self.obs, self.step_reward, self.done, self.info = self.env.step(self.actions)
            print(self.actions)

            self.camera_follow(old_cam_point)
            laptime += self.step_reward
            self.env.render(mode='human_fast')

            if self.env.lap_counts[0] == 1:
                break;

        self.end_sim(laptime, ftg_laptime, start)

def my_try(z, x):
    return x*2


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
    cars = []
    for i in range(num):
        cars.append(Car(MPC(), GapFollower()))

    pool = mp.Pool(mp.cpu_count())
    runner = GymRunner(cars, map_obstacles)
    runner.run(zoom)
    pool.close()
    pool.join()

