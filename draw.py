# draw_line.py

import random

from PIL import Image, ImageDraw


def draw_obstacle(image_path, output_path, point, size=15, div_x=0.5):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    div_y = 1 - div_x
    draw.ellipse(
        (point[0] - (size * div_x), point[1] - (size * div_x), point[0] + (size * div_y), point[1] + (size * div_y)),
        fill="black", outline="black")

    image.save(output_path)


def un_convert(point):
    map_resolution = 0.05796
    origin = [-84.85359914210505, -36.30299725862132]
    point[0] = (point[0] - origin[0]) / map_resolution
    point[1] = 2000 - (point[1] - origin[1]) / map_resolution

    return point


def add_obstacles(obstacles):
    orig_map = "map/Spielberg_map.png"
    new_map = "obstacle_map/new_spielberg.png"

    for obstacle in obstacles:
        obs = un_convert(obstacle)
        draw_obstacle(orig_map, new_map, obs)
        orig_map = new_map


