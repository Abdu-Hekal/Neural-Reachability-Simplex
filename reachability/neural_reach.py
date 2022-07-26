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
    return x


def get_reachset(inputs_list, models):
    """
    Compute untransformed reachable set based on input and bloat the output to bias for overapproximation
   """
    sf_list = []
    # test data for initial region as shown in JuliaReach documentation
    x = inputs_list

    for i in range(8):
        rnn_forward = models[i][0].forward(x.float().view(1, len(x), 12))
        val = models[i][1].forward(rnn_forward[0])
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

    rnn_forward = model[0].forward(x.float().view(1, len(x), 12))
    vals = model[1].forward(rnn_forward[0])
    vals = vals.tolist()

    return vals


def get_theta_max_list(inputs_list, model_optim):
    """
   Get list of maximum orientations for each reach image
   """
    # test data for initial region as shown in JuliaReach documentation
    x = inputs_list

    rnn_forward = model_optim[0].forward(x.float().view(1, len(x), 12))
    vals = model_optim[1].forward(rnn_forward[0])
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
