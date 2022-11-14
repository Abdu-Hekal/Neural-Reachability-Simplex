"""

Generating untransformed reachable sets from trained neural networks using control inputs and initial sets

author: Abdelrahman Hekal

"""
import torch
from pytope import Polytope

# CUDA for PyTorch
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def load_checkpoint(filepath):
    """
   Load trained neural network
   """
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    # no retraining
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval().to(device)

    return model


def get_models():
    """
   Get a list of the eight trained neural network corresponding to each template direction
   """
    models = []
    for i in range(8):
        file = 'reachability/bicyclemodels/0.5s_mpc/0.5s_mpc_dir' + str(i + 1) + '_model.pth'
        nn = load_checkpoint(file)
        models.append(nn)
    return models


def gpu_inputs_list(inputs_list):
    """
   Get input list and standardize it
   """
    x = torch.tensor(inputs_list).to(device)
    return x


def get_reachset(inputs_list, models):
    """
    Compute untransformed reachable set based on input and bloat the output to bias for overapproximation
   """
    sf_list = []
    for model in models:
        val = get_vals(inputs_list, model)
        val = val.add(0.1)  # bloating
        val = val.tolist()
        sf_list.append(val)

    return sf_list


def sf_to_poly(sf):
    """
   Combine computed support value with corresponding support directions and store as a polytope
   """
    A = [
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
        [1, 0]]

    b = sf
    Poly = Polytope(A, b)

    return Poly


def get_vals(inputs_list, model):
    """
   Get list of minimum orientations for each reach image
   """
    x = inputs_list

    rnn_forward = model[0].forward(x.float().view(1, len(x), 12))
    vals = model[1].forward(rnn_forward[0])

    return vals


def get_theta_min_model():
    file = 'reachability/bicyclemodels/0.5s_mpc/0.5s_mpc_theta_min_model.pth'
    model_optim = load_checkpoint(file)
    return model_optim


def get_theta_max_model():
    file = 'reachability/bicyclemodels/0.5s_mpc/0.5s_mpc_theta_max_model.pth'
    model_optim = load_checkpoint(file)
    return model_optim


def get_vel_min_model():
    file = 'reachability/bicyclemodels/0.5s_mpc/0.5s_mpc_v_min_model.pth'
    model_optim = load_checkpoint(file)
    return model_optim


def get_vel_max_model():
    file = 'reachability/bicyclemodels/0.5s_mpc/0.5s_mpc_v_max_model.pth'
    model_optim = load_checkpoint(file)
    return model_optim
