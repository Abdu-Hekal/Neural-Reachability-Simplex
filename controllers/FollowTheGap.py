import time
import gym
import numpy as np
import concurrent.futures
import yaml
from argparse import Namespace
from pyglet import shapes
import reachset_transform.main as transform

from shapely.geometry import Polygon as shapely_poly



# import your drivers here
from drivers import GapFollower

# choose your drivers here (1-4)
drivers = [GapFollower()]

# choose your racetrack here (SOCHI, SOCHI_OBS)
RACETRACK = 'Spielberg'


class GymRunner(object):

    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self):
        # load map

        with open('../obstacle_map/new_config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]
                                                           ]))
        env.render()

        laptime = 0.0
        start = time.time()

        while not done:
            actions = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, driver in enumerate(drivers):
                    futures.append(executor.submit(driver.process_lidar, obs['scans'][i]))
            for future in futures:
                speed, steer = future.result()
                actions.append([steer, speed])
            actions = np.array(actions)
            obs, step_reward, done, info = env.step(actions)
            laptime += step_reward
            env.render(mode='human')

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    runner = GymRunner(RACETRACK, drivers)
    runner.run()
