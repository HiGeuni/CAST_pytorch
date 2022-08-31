import torch
from models import net

vgg = net.vgg
vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))

print(vgg)