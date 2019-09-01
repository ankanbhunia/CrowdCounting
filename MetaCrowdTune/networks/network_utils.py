import numpy as np
import torch
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)


def forward_pass(network, _in, _tar, mode='validation', weights=None):
    _input = _in.cuda()
    _input = Variable(_input)

    _target = _tar.type(torch.FloatTensor).unsqueeze(0).cuda()
    _target = Variable(_target)

    output = network.network_forward(_input, _target, weights)

    if mode == 'validation':
        return [output]
    else:
        loss = network.loss
        return [output, loss]


def evaluate_(network, dataloader, mode='validation', weights=None):
    mae, mse, loss = 0.0, 0.0, 0.0
    for idx, (_in, _tar) in enumerate(dataloader):
        # batch_size = _in.numpy().shape[0]
        result = forward_pass(network, _in, _tar, mode, weights)
        difference = result[0].data.sum() - _tar.sum().type(torch.FloatTensor).cuda()
        _mae = torch.abs(difference)
        _mse = difference ** 2

        # mae += (abs(difference).item())
        # mse += (difference.item() ** 2)
        mae += _mae.item()
        mse += _mse.item()

        if mode == 'training':
            loss += result[1].item()

    mae /= len(dataloader)
    mse = np.sqrt(mse / len(dataloader))

    if mode == 'training':
        loss /= len(dataloader)
        return (loss, mae, mse)

    return mae, mse


def evaluate(network, dataloader, mode='validation', weights=None):
    mae, mse, loss = 0.0, 0.0, 0.0
    for idx, (_in, _tar) in enumerate(dataloader):
        # batch_size = _in.numpy().shape[0]
        

        _input = _in.cuda()
        _input = Variable(_input)

        _target = _tar.type(torch.FloatTensor).unsqueeze(0).cuda()
        _target = Variable(_target)

        result = network(_input, _target)
        
        difference = result.data.sum() - _tar.sum().type(torch.FloatTensor).cuda()
        _mae = torch.abs(difference)
        _mse = difference ** 2

        # mae += (abs(difference).item())
        # mse += (difference.item() ** 2)
        mae += _mae.item()
        mse += _mse.item()
        loss_ = network.loss
        if mode == 'training':
            loss += loss_.item()

    mae /= len(dataloader)
    mse = np.sqrt(mse / len(dataloader))

    if mode == 'training':
        loss /= len(dataloader)
        return (loss, mae, mse)

    return mae, mse
