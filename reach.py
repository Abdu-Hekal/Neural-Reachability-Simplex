"""

Neural Network Reachability

author: Abdelrahman Hekal


"""
import pyglet
from pyglet.graphics import TextureGroup

import reachability.main as reach
import reachset_transform.main as transform
import Car_dimensions.main as car_reach

from shapely.geometry import Polygon as shapely_poly
from shapely.geometry import Point


models = reach.get_models()
theta_min_model = reach.get_theta_min_model()
theta_max_model = reach.get_theta_max_model()


def reachability(oa, odelta, state, batch, map_obstacles):
    """
    Generate reachable sets and check for intersection with obstacle

   """

    input_list = []
    intersect = False
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    for i in range(60):
        nn_input = [i + 1, oa[0], odelta[0], oa[1], odelta[1], oa[2], odelta[2], oa[3], odelta[3], oa[4],
                    odelta[4],
                    state.v]
        input_list.append(nn_input)
    gpu_input_list = reach.gpu_inputs_list(input_list)
    sf_list = reach.get_reachset(gpu_input_list, models)
    theta_min_list = reach.get_theta_min_list(gpu_input_list, theta_min_model)
    theta_max_list = reach.get_theta_max_list(gpu_input_list, theta_max_model)

    vertices_list = []
    for reach_iter in range(60):  # range(99, -1, -1)
        new_sf_list = []
        for dirs in range(len(sf_list)):
            new_sf_list.append(sf_list[dirs][0][reach_iter][0])
        poly_reach = reach.sf_to_poly(new_sf_list)

        full_reach_poly = car_reach.add_car_to_reachset(poly_reach, theta_min_list[0][reach_iter][0],
                                                        theta_max_list[0][reach_iter][0])

        final_reach_poly = transform.transform_poly(full_reach_poly, state.yaw, state.x, state.y)

        # checks intersection with reachset and obstacle
        if reach_iter < 30:
            reachpoly = shapely_poly(final_reach_poly.V)
            for map_obstacle in map_obstacles:
                p = Point(map_obstacle)
                c = p.buffer(0.75).boundary
                if c.intersects(reachpoly):
                    intersect = True

        vertices = []
        for vertex in final_reach_poly.V:
            for val in vertex:
                vertices.append(50 * val)
        background = pyglet.graphics.OrderedGroup(-1)
        howmany = int(len(vertices)/2)
        colors = (128,255,0, 20)*howmany
        vertex_list = batch.add(len(final_reach_poly.V), pyglet.gl.GL_POLYGON, background, ('v2f', vertices),  ('c4B', colors))
        vertices_list.append(vertex_list)

    return vertices_list, intersect
