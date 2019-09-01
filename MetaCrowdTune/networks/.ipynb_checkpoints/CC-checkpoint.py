import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torchvision import models
from gumbel_softmax import gumbel_softmax
from SCC_Model.Res50_ import resnet26 as net
from Policy import resnet
from config import cfg

def Load_Weights_from_ImageNet(MODEL):

    resnet50 = models.resnet50(pretrained=True)
    parameter_of_model_A = resnet50.state_dict()
    parameter_of_model_B = MODEL.state_dict()

    def map_(name):
        mapping = {'series_blocks.0': 'layer1', 'series_blocks.1': 'layer2', 'series_blocks.2': 'layer3',
                   'parallel_blocks.0': 'layer1', 'parallel_blocks.1': 'layer2', 'parallel_blocks.2': 'layer3'}
        for i in mapping.keys():
            name = name.replace(i, mapping[i])
        return name

    k = {i: parameter_of_model_A[map_(i)] for i in parameter_of_model_B.keys() if parameter_of_model_A.has_key(map_(i))}

    MODEL.load_state_dict(k, strict=False)

    return MODEL

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        
        self.net = net()
        self.net = Load_Weights_from_ImageNet(self.net)
        
        #self.no_of_layers = sum(self.net.layer_config)
        #self.agent = resnet(self.no_of_layers * 2)

       
        #for name, m in self.net.named_modules():
        #    if 'series_blocks' in name:
        #        try:
        #            m.weight.requires_grad = False
        #        except:
        #            pass
   
    def forward(self,x):

        if cfg.Mode == 'WithPolilcy':
            probs = self.agent(x)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            #print (action)
            policy = action[:, :, 1]
            y = self.net(x, policy)
            return y

        else:
            h = self.net(x)
            print (h.shape)
            return  h

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        

        if model_name == 'AlexNet':
            from SCC_Model.AlexNet import AlexNet as net        
        elif model_name == 'VGG':
            from SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from SCC_Model.Res50_ import resnet26 as net
        elif model_name == 'Res101':
            from SCC_Model.Res101 import Res101 as net            
        elif model_name == 'Res101_SFCN':
            from SCC_Model.Res101_SFCN import Res101_SFCN as net

        self.CCN = Network()
        #self.CCN = Load_Weights_from_ImageNet(self.CCN)

        self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    def copy_weights(self, network):
        for m_from, m_to in zip(network.modules(), self.modules()):
            if isinstance(m_to, nn.Conv2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map, weights = None):     

        if weights is not None:
            density_map = self.CCN(img) 
        else:
            #self.CNN.load_state_dict(weights)
            density_map = self.CCN(img) 
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    
    def build_loss(self, density_map, gt_data):
        print density_map.shape, gt_data.shape
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

