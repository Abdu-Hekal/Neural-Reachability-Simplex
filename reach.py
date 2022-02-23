"""

Neural Network Reachability

author: Abdelrahman Hekal


"""

import reachability.main as reach
import reachset_transform.main as transform
import Car_dimensions.main as car_reach


def reachability(oa, odelta, state, batch):
    """
    Generate reachable sets and check for intersection with obstacle

   """

    vertix_lists = []

    input_list = []
    for i in range(100):
        nn_input = [i + 1, oa[0], odelta[0], oa[1], odelta[1], oa[2], odelta[2], oa[3], odelta[3], oa[4],
                    odelta[4],
                    state.v]
        input_list.append(nn_input)
    gpu_input_list = reach.gpu_inputs_list(input_list)
    sf_list = reach.get_reachset(gpu_input_list, models)
    theta_min_list = reach.get_theta_min_list(gpu_input_list, theta_min_model)
    theta_max_list = reach.get_theta_max_list(gpu_input_list, theta_max_model)

    for reach_iter in range(100):  # range(99, -1, -1)
        new_sf_list = []
        for dirs in range(len(sf_list)):
            new_sf_list.append(sf_list[dirs][0][reach_iter][0])
        poly_reach = reach.sf_to_poly(new_sf_list)

        full_reach_poly = car_reach.add_car_to_reachset(poly_reach, theta_min_list[0][reach_iter][0],
                                                        theta_max_list[0][reach_iter][0])
        final_reach_poly = transform.transform_poly(full_reach_poly, state.yaw, state.x, state.y)

        # final_reach_poly.V : this is probably the vertices
        print(final_reach_poly.V)

    return vertix_lists



