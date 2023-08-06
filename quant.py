import torch
from torch import nn
from torchvision.models import resnet50
from torchinfo import summary
from matplotlib import pyplot as plt
from pathlib import Path

def load_model(model:nn.Module, checkpint:str | Path, num_classes=10):
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    model.load_state_dict(torch.load(checkpint)['net'])

def plot_weight_dist(model:nn.Module):
    plot_idx = 1
    for idx, (n, m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            plt.subplot(4,4,plot_idx)
            plt.hist(m.weight.data.flatten(), bins=100)
            plt.title(n)
            plot_idx += 1
        if plot_idx > 16:
            plt.show()
            plot_idx = 1
    plt.show()

def quant_conv_weight(model:nn.Module, kbit:int=4, symmetric:bool=True):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # quantize weight
            max_value = m.weight.data.abs().max()
            m.weight.data = torch.clip(torch.round(m.weight.data/max_value*(2**kbit-1), decimals=0), -2**kbit, 2**kbit-1)
            
            # dequantize weight
            m.weight.data = m.weight.data/(2**kbit-1) * max_value
    
if __name__=='__main__':
    model = resnet50()
    load_model(model, './checkpoint/ckpt.pth', 10)
    summary(model, (1,3,224,224))
    
    quant_conv_weight(model)
    plot_weight_dist(model)