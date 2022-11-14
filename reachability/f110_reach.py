"""

Neural Network Reachability

author: Abdelrahman Hekal


"""
import pyglet
import reachability.neural_reach as reach
import reachset_transform.main as transform
import Car_dimensions.main as car_reach

models = reach.get_models()
theta_min_model = reach.get_theta_min_model()
theta_max_model = reach.get_theta_max_model()
vel_min_model = reach.get_vel_min_model()
vel_max_model = reach.get_vel_max_model()


def reachability(state, oa, odelta, batch=None, color=None, car_num=0, time_horizon_min=0, time_horizon_max=0.5,
                 step_size=0.01, plot=True):
    """
    Generate reachable sets and plot them

   """

    sf_list, theta_min_list, theta_max_list = compute_reachsets(oa, odelta, state, time_horizon_min, time_horizon_max,
                                                                step_size)

    vertices_list = []
    polys = []
    for reach_iter in range(int(time_horizon_max / step_size) - int(time_horizon_min / step_size)):  # range(99, -1, -1)
        final_reach_poly = transform_reachsets(reach_iter, sf_list, theta_min_list[0][reach_iter][0],
                                               theta_max_list[0][reach_iter][0], state)

        if final_reach_poly:
            polys.append(final_reach_poly)
            if plot:
                plot_reachset(car_num, final_reach_poly, batch, vertices_list, color)

    return vertices_list, polys


def compute_velocity(oa, odelta, state, time_horizon_min=0, time_horizon_max=0.5, step_size=0.01):
    gpu_input_list = get_nn_input_list(oa, odelta, state, time_horizon_min, time_horizon_max, step_size)
    vel_min_list = reach.get_vals(gpu_input_list, vel_min_model).tolist()
    vel_max_list = reach.get_vals(gpu_input_list, vel_max_model).tolist()

    return vel_min_list, vel_max_list


def compute_reachsets(oa, odelta, state, time_horizon_min=0, time_horizon_max=0.5, step_size=0.01):
    gpu_input_list = get_nn_input_list(oa, odelta, state, time_horizon_min, time_horizon_max, step_size)
    sf_list = reach.get_reachset(gpu_input_list, models)
    theta_min_list = reach.get_vals(gpu_input_list, theta_min_model).tolist()
    theta_max_list = reach.get_vals(gpu_input_list, theta_max_model).tolist()

    return sf_list, theta_min_list, theta_max_list


def get_nn_input_list(oa, odelta, state, time_horizon_min=0, time_horizon_max=0.5, step_size=0.01):
    input_list = []
    for i in range(int(time_horizon_min / step_size), int(time_horizon_max / step_size)):
        nn_input = [i + 1, oa[0], odelta[0], oa[1], odelta[1], oa[2], odelta[2], oa[3], odelta[3], oa[4],
                    odelta[4],
                    state.v]
        input_list.append(nn_input)
    gpu_input_list = reach.gpu_inputs_list(input_list)

    return gpu_input_list


def transform_reachsets(reach_iter, sf_list, theta_min, theta_max, state):
    new_sf_list = []
    for dirs in range(len(sf_list)):
        new_sf_list.append(sf_list[dirs][0][reach_iter][0])
    poly_reach = reach.sf_to_poly(new_sf_list)

    if poly_reach.V.size != 0:
        full_reach_poly = car_reach.add_car_to_reachset(poly_reach, theta_min, theta_max)
        final_reach_poly = transform.transform_poly(full_reach_poly, state.yaw, state.x, state.y)
    else:
        final_reach_poly = None

    return final_reach_poly


def plot_reachset(car_num, final_reach_poly, batch, vertices_list, color):
    vertices = []
    for vertex in final_reach_poly.V:
        for val in vertex:
            vertices.append(50 * val)

    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    background = pyglet.graphics.OrderedGroup(-(car_num + 1))
    howmany = int(len(vertices) / 2)
    colors = color * howmany

    vertex_list = batch.add(len(final_reach_poly.V), pyglet.gl.GL_POLYGON, background, ('v2f', vertices),
                            ('c4B', colors))
    vertices_list.append(vertex_list)
