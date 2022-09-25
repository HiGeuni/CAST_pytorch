import torch
import torch.nn as nn
from models import MSP
from models.cast_model import init_net
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import os
import glob
import json

to_tensor = ToTensor()
t = [64, 128, 256, 512]

genre_cnt = {}

# hyper parameter
gpu_ids = [1]
new_shape = (256, 256)
img_root_dir = "datasets/test_0807/trainB/"

style_vgg_path = 'models/style_vgg.pth'
model_checkpoint = './checkpoints/CAST_model_0810'

# The Function that load MSP network
# require models
def load_netDP(style_vgg_path, gpu_ids, model_checkpoint):

    load_filename = [f'latest_net_{i}.pth'  for i in ['D', 'P_style']]
    load_path = [os.path.join(model_checkpoint, load_filename[i]) for i in range(2)]

    style_vgg = MSP.vgg
    style_vgg.load_state_dict(torch.load(style_vgg_path))
    style_vgg = nn.Sequential(*list(style_vgg.children()))

    net = []
    netD = MSP.StyleExtractor(style_vgg, gpu_ids)
    netP_style = MSP.Projector(gpu_ids)
    init_net(netD, 'normal', 0.02, gpu_ids) 
    init_net(netP_style, 'normal', 0.02, gpu_ids)
    net.append(netD) 
    net.append(netP_style)

    # load model
    for i in range(2):
        state_dict = torch.load(load_path[i], map_location='cpu')
        net[i].load_state_dict(state_dict)

    return net

def loadImgTo4DTensor(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(new_shape)
    img = to_tensor(img)
    img = img.unsqueeze(0).to("cuda:1")
    return img

# Get one file per one theme
def get_file_per_theme(path):
    pat = glob.glob(path)
    dc = {}
    ls = []
    for i in pat:
        genre = i.split('-')[0].split('/')[-1].split('_')[0]
        if genre not in dc:
            dc[genre] = 1
            ls.append(i)
    return ls

# get all file list which include pat(theme)
def get_file_list_all(pat):
    pat2 = glob("/home/khyo/wikiart/train_for/*")
    ls = []
    for i in pat:
        genre = i.split('-')[0].split('/')[-1].split('_')[0]
        genre_cnt[genre] += 1
        ls.append(i)
    return ls

# Get List of 27 Theme in specific path
def get_theme_list(path = None):
    pat = glob.glob("/home/khyo/wikiart/wikiart/*")
    dc = {}
    for i in pat:
        if '.' in i:
            continue

        genre = i.split('-')[0].split('/')[-1].split('_')[0]
        
        if genre not in dc:
            genre_cnt[genre] = 0
            dc[genre] = 1
    return list(dc.keys())

def get_author_list():
    pat = glob.glob("/home/khyo/wikiart/train_for/*")
    dc = {}
    for i in pat:
        author = i.split('/')[-1].split('_')[-2]

        if author not in dc:
            dc[author] = 1
    return list(dc.keys())

def get_directory_file_list(path):
    pat = glob.glob(path)
    return pat

def get_weight():
    with open("theme_weights.json", "r") as f:
        data = json.load(f)
    return data
    
def diff(a, b):
    tmp = 0
    for i in range(4):
        for j in range(t[i]):
            tmp += abs(a[i][j] - b[i][j])
    return tmp

def get_code(net, img_path):
    cnt = 0
    tt = []
    ls = [0 for _ in range(960)]

    img = loadImgTo4DTensor(img_path)

    code = net[0].forward(img, [0,1,2,3])

    for j in code:
        tt.append(torch.flatten(j).tolist())

    for j in range(4):
        for num in range(t[j]):
            ls[cnt] += tt[j][num]
            cnt += 1

    return ls

def get_genre_weight():
    genre = get_genre_list()

    res = {}

    # i : 장르 이름
    for i in genre:
        image = get_file_list_all(i)
        
        exp = [ 0 for _ in range(960) ]
        ls = [0 for _ in range(960)]
        var = [ 0 for _ in range(960)]
        codes = [[] for _ in range(len(image))] 

        for k in range(len(image)):
            img = loadImgTo4DTensor(image[k])
            code = net[0].forward(img,[0, 1, 2, 3])
            for j in code:
                codes[k].append(torch.flatten(j).tolist())
            cnt = 0
            for j in range(4):
                for num in range(t[j]):
                    ls[cnt] += codes[k][j][num]
                    cnt += 1

        # 각 위치의 평균을 구한다.
        for k in range(960):
            exp[k] = ls[k] / genre_cnt[i]
        
        for k in range(960):
            var[k] += pow(ls[k] - exp[k], 2)

        
        res[i] = ls

    with open("theme_weights.json", "w") as json_file:
        json.dump(res, json_file)

def getCsv():
    genre = get_genre_list()
    res = {}


# net = load_netDP(style_vgg_path, gpu_ids, model_checkpoint)

# files = get_directory_file_list("./datasets/test_0718/testB/*")
# # files = get_directory_file_list("./datasets/test_0807/trainB/*")

# data = get_weight()

# theme_list = data.keys()

# ans_cnt = 0

# for f in files:
#     if f[-4:] == "webp":
#         continue
#     ls = get_code(f)
#     theme = f.split('/')[-1].split('.')[0]
#     min_ = 2e9
#     gen = ""

#     for key in data.keys():
#         tmp = 0
#         for i in range(960):
#             tmp += abs(ls[i] - data[key][i])
#         if(tmp < min_):
#             min_ = tmp
#             gen = key
#     if gen in f:
#         ans_cnt += 1
#     print(theme, gen)

# print(ans_cnt)