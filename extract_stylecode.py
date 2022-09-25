import utils
import pandas as pd
import numpy as np
from glob import glob

model_checkpoint = './checkpoints/CAST_model_0810'

net = utils.load_netDP(utils.style_vgg_path, [1], model_checkpoint)

file_list = glob("/home/khyo/wikiart/train_for/*")

for name in file_list[:5]:
    print(name.split('/')[-1].split('_')[-2])

df = pd.DataFrame(columns = ["file_path", *range(960)])

cnt = 0

for image in file_list:
    code = utils.get_code(net, image)
    df.loc[cnt] = [image, *code]
    cnt += 1

df.to_csv("./weights.csv")