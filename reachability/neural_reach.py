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
print(device)


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
        file = 'reachability/bicyclemodels/new_bicycle_dir' + str(i + 1) + '_100kSamples_100kEpochs.pth'
        nn = load_checkpoint(file)
        models.append(nn)
    return models


def gpu_inputs_list(inputs_list):
    """
   Get input list and standardize it
   """
    x = torch.tensor(inputs_list).to(device)

    x_mean = 4.5028
    x_std = 16.2052

    x = (x - x_mean) / x_std

    return x


def get_reachset(inputs_list, models):
    """
    Compute untransformed reachable set based on input and bloat the output to bias for overapproximation
   """
    sf_list = []
    # test data for initial region as shown in JuliaReach documentation
    x = inputs_list

    y_std = [2.5071, 1.1776, 1.9493, 1.7887, 2.0338, 1.1589, 2.4328, 2.0179]
    y_mean = [2.6990, 0.2936, -1.8478, -2.0807, -1.8901, 0.2159, 2.5820, 2.4274]

    for i in range(8):
        rnn_forward = models[i][0].forward(x.float().view(1, len(x), 12))
        val = models[i][1].forward(rnn_forward[0])
        val = (val * y_std[i]) + y_mean[i]
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


def get_theta_min_list(inputs_list, model):
    """
   Get list of minimum orientations for each reach image
   """
    x = inputs_list

    y_std = 0.4052
    y_mean = -0.0478

    rnn_forward = model[0].forward(x.float().view(1, len(x), 12))
    val = model[1].forward(rnn_forward[0])
    vals = (val * y_std) + y_mean
    vals = vals.tolist()

    return vals


def get_theta_max_list(inputs_list, model_optim):
    """
   Get list of maximum orientations for each reach image
   """
    # test data for initial region as shown in JuliaReach documentation
    x = inputs_list

    y_std = 0.4274
    y_mean = 0.1595

    rnn_forward = model_optim[0].forward(x.float().view(1, len(x), 12))
    val = model_optim[1].forward(rnn_forward[0])
    vals = (val * y_std) + y_mean
    vals = vals.tolist()

    return vals


def get_theta_min_model():
    file = 'reachability/bicyclemodels/new_bicycle_thetaMin_100kSamples_100kEpochs.pth'
    model_optim = load_checkpoint(file)
    return model_optim


def get_theta_max_model():
    file = 'reachability/bicyclemodels/new_bicycle_thetaMax_100kSamples_100kEpochs.pth'
    model_optim = load_checkpoint(file)
    return model_optim