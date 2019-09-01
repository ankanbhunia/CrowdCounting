import logging
from collections import OrderedDict

from datasets.dataloader_main import GetDataLoader
from CC import CrowdCounter
from network_utils import *

torch.manual_seed(1)


class BaseNetwork(CrowdCounter):
    def __init__(self, loss_function, base_updates, base_lr, base_batch, meta_batch, num_of_channels=3):
        super(BaseNetwork, self).__init__(loss_function, num_of_channels)

        self.loss_function = loss_function
        self.base_updates = base_updates
        self.base_lr = base_lr
        self.base_batch = base_batch
        self.meta_batch = meta_batch
        self.num_of_channels = num_of_channels
        self.get_loader = GetDataLoader()



    def network_forward(self, x, target_, weights=None):
        return super(BaseNetwork, self).forward(x,target_, weights)

    def forward_pass(self, _input, _output, weights=None):
        _input = torch.autograd.Variable(_input.cuda())
        _target = torch.autograd.Variable(_output.type(torch.FloatTensor).cuda())

        output = self.network_forward(_input, _target, weights)
        loss = self.loss
        return loss, output

    def forward(self, task):
        def mul(a,b):
            if b is None:
                return torch.tensor(np.zeros_like(a))
            else:
                return a*b
        train_loader = self.get_loader.get_data(task)
        validation_loader = self.get_loader.get_data(task, mode='validation')

        # testing the base network before training
        train_pre_loss, train_pre_accuracy, train_pre_mse = evaluate_(self, train_loader, mode='training')
        validation_pre_accuracy, validation_pre_mse = evaluate_(self, validation_loader)

        base_weights = OrderedDict(
            (name, parameter) for (name, parameter) in self.named_parameters() if parameter.requires_grad)

        for idx, data in enumerate(train_loader):

            _input, _target = data
            _input = Variable(_input.cuda())

            _target = Variable(_target.type(torch.FloatTensor).unsqueeze(0).cuda())

            if idx == 0:
                trainable_weights = [p for n, p in self.named_parameters() if p.requires_grad]
                loss, _ = self.forward_pass(_input, _target)
                gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True, allow_unused = True)
            else:
                trainable_weights = [v for k, v in base_weights.items() if 'frontend' not in k]
                #loss, _ = self.forward_pass(_input, _target, base_weights)
                loss, _ = self.forward_pass(_input, _target)
                gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True, allow_unused = True)
    
    

            
            base_weights = OrderedDict((name, parameter - mul(self.base_lr, gradient)) for ((name, parameter), gradient) in
                                       zip(base_weights.items(), gradients) if parameter.requires_grad)
            
            load_weights(base_weights)

        # testing the base network after training to evaluate fast adaptation
        train_post_loss, train_post_accuracy, train_post_mse = evaluate_(self, train_loader, mode='training',
                                                                        weights=base_weights)
        validation_post_accuracy, validation_pose_mse = evaluate_(self, validation_loader,
                                                                 weights=base_weights)

        logging.info("==========================")
        logging.info("(Meta-training) pre train loss: {}, MAE: {}, MSE: {}".format(train_pre_loss, train_pre_accuracy,
                                                                                   train_pre_mse))
        logging.info(
            "(Meta-training) post train loss: {}, MAE: {}, MSE: {}".format(train_post_loss, train_post_accuracy,
                                                                           train_post_mse))
        logging.info(
            "(Meta-training) pre-training test MAE: {}, MSE: {}".format(validation_pre_accuracy, validation_pre_mse))
        logging.info(
            "(Meta-training) post-training test MAE: {}, MSE: {}".format(validation_post_accuracy, validation_pose_mse))
        logging.info("==========================")

        # updating the meta network with the accumulated gradients from training base network
        # this operation is performed by running a dummy forward pass through the meta network on the validation dataset

        _input, _target = validation_loader.__iter__().next()
        _target = Variable(_target.type(torch.FloatTensor).unsqueeze(0).cuda())
        loss, _ = self.forward_pass(_input, _target, base_weights)
        loss /= self.meta_batch
        trainable_weights = {n: p for n, p in self.named_parameters() if p.requires_grad}
        gradients = torch.autograd.grad(loss, trainable_weights.values(), allow_unused=True)
        meta_gradients = {name: grad for ((name, _), grad) in zip(trainable_weights.items(), gradients)}
        metrics = (train_post_loss, train_post_accuracy, train_post_mse, validation_post_accuracy, validation_pose_mse)
        return metrics, meta_gradients
