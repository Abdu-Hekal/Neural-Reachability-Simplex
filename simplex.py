import random
import time
import gym
import numpy as np
import pyglet
import yaml
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from car import Car
from controllers.mpc import MPC
from map.start_point import get_rand_start_point
from obstacle_map import draw
from controllers.drivers import GapFollower
import reachability.f110_reach as reach
from shapely.geometry import Point
from shapely.geometry import Polygon as shapely_poly
import multiprocessing as mp
import copy


class GymRunner(object):

    def __init__(self, cars, map_obstacles):
        self.cars = cars
        self.map_obstacles = []
        self.set_map_obstacles(map_obstacles)
        self.setup_env()
        self.setup_mpcs()
        self.setup_start_points()
        self.setup_colors()
        self.actions = None

    def set_map_obstacles(self, map_obstacles):
        for map_obstacle in map_obstacles:
            p = Point(map_obstacle)
            c = p.buffer(0.75).boundary
            self.map_obstacles.append(c)

    def setup_env(self):
        with open('obstacle_map/new_config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)
        self.env = gym.make('f110_gym:f110-v0', map=self.conf.map_path, map_ext=self.conf.map_ext,
                            num_agents=len(self.cars))

    def setup_start_points(self):
        start_points = get_rand_start_point(self.conf.wpt_path, len(self.cars))
        env_array = []
        for start_point, car in zip(start_points, self.cars):
            index, point = start_point
            point_array = point.split(";")
            env_array.append([float(point_array[1]), float(point_array[2]), float(point_array[3])])
            car.advanced_controller.controller.init_target_ind = index
        self.obs, self.step_reward, self.done, self.info = self.env.reset(np.array(env_array))
        self.env.render()

    def setup_mpcs(self):
        for car in cars:
            car.advanced_controller.setup(self.conf, self.env, car)

    def setup_colors(self):
        self.colors = []
        for i in range(len(self.cars)):
            self.colors.append((random.randint(0, 250), random.randint(0, 250), random.randint(0, 250), 10))

    def select_control(self):
        actions = []
        for index, (car, reach_color) in enumerate(zip(self.cars, self.colors)):
            if car.control_count == 10:
                self.check_intersection(car, index)
                car.rm_plotted_reach_sets()
                car.baseline = False
                car.label = "MPC"
                speed, steer, state = car.advanced_controller.step(self.obs)
                car.action = [steer, speed]
                vertices_list, polys = reach.reachability(state, car.advanced_controller.controller.oa,
                                                          car.advanced_controller.controller.odelta,
                                                          self.env.renderer.batch, reach_color,
                                                          car.advanced_controller.num)
                car.vertices_list += vertices_list
                car.reachset = polys
                car.control_count = 0
            if car.intersect or car.baseline:
                car.baseline = True
                car.label = "FTG"
                speed, steer, state = car.baseline_controller.process_lidar(self.obs['scans'][index],
                                                                            MAX_SPEED=car.MAX_SPEED)
                car.action = [steer, speed]
                car.baseline_laptime += self.step_reward

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

    def check_intersection(self, car, car_num):
        car.intersect = False
        # result_objects = [pool.apply_async(self.check_one_poly_intersection, (final_reach_poly, poly_index, car_num)) for poly_index, final_reach_poly in enumerate(car.reachset)]
        # intersect = [r.get() for r in result_objects]
        intersect = []
        for poly_index, final_reach_poly in enumerate(car.reachset):
            intersect.append(self.check_one_poly_intersection(final_reach_poly, poly_index, car_num))
        car.intersect = any(intersect)

    def check_one_poly_intersection(self, final_reach_poly, poly_index, car_num):
        intersect = False
        reachpoly = shapely_poly(final_reach_poly.V)
        min_other_car = max(0, poly_index)
        max_other_car = min(50, poly_index + 1)
        # check intersection with obstacles
        for map_obstacle in self.map_obstacles:
            if reachpoly.intersects(map_obstacle):
                intersect = True
        # check intersection of vehicles with one another
        for x in range(len(self.cars) - 1 - car_num):
            other_car = self.cars[-(x + 1)]
            for other_reachpoly in other_car.reachset[min_other_car:max_other_car]:
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

    def end_sim(self, car_num, laptime, ftg_laptime, start):
        print(f"car {car_num} follow the gap control: {round((ftg_laptime / laptime) * 100, 3)}%")
        print(f"car {car_num} mpc control: {100 - round(((ftg_laptime / laptime) * 100), 3)}%")
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

    def run(self, zoom):
        # load map
        self.env.renderer.set_fullscreen(True)
        self.env.renderer.set_mouse_visible(False)
        start = time.time()
        laptime = 0

        pyglet_label = pyglet.text.Label('{label}'.format(
            label='MPC'), font_size=200, x=1000, y=1800, anchor_x='center', anchor_y='center',
            color=(255, 255, 255, 255), batch=self.env.renderer.batch)

        while not self.done and self.env.renderer.alive:
            start = time.time()

            self.check_zoom(zoom)
            self.select_control()
            print(f"full time: {time.time() - start} seconds")

            old_cam_point = [self.obs['poses_x'][0], self.obs['poses_y'][0]]
            pyglet_label.text = ""
            for i, car in enumerate(self.cars):
                pyglet_label.text += f"{car.label}, "
            self.obs, self.step_reward, self.done, self.info = self.env.step(self.actions)

            self.camera_follow(old_cam_point)
            laptime += self.step_reward
            self.env.render(mode='human_fast')

            # if self.env.lap_counts[0] == 1:
            #     break;

        for i, car in enumerate(self.cars):
            self.end_sim(i + 1, laptime, car.baseline_laptime, start)


def my_try(z, x):
    return x * 2


if __name__ == '__main__':
    parser = ArgumentParser(description="Settings",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-z", "--zoom", action="store_true", help="Zoom in camera")
    parser.add_argument("-n", "--number", type=int, default=1, help='number of vehicles')
    parser.add_argument("-o", "--obstacles", type=int, default=5, help='number of obstacles')
    args = vars(parser.parse_args())
    zoom = args['zoom']
    num = args['number']
    num_obstacles = args['obstacles']
    rand_points = get_rand_start_point('map/Spielberg_centerline.csv', num_obstacles)
    map_obstacles = []
    for rand_point in rand_points:
        index, point = rand_point
        point_array = point.split(",")
        map_obstacles.append([float(point_array[0]), float(point_array[1])])
    draw.add_obstacles(copy.deepcopy(map_obstacles))

    cars = []
    for i in range(num):
        car = Car(i, MPC(), GapFollower())
        car.MAX_SPEED = random.uniform(3, 5)
        print(f"car {i + 1} max speed is {car.MAX_SPEED} m/s")
        cars.append(car)

    pool = mp.Pool(mp.cpu_count())
    runner = GymRunner(cars, map_obstacles)
    runner.run(zoom)
    pool.close()
    pool.join()
