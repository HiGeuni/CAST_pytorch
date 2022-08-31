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

load_dir = './checkpoints/CAST_model_0810'
model_names = ['D', 'P_style']
load_filename = [f'latest_net_{i}.pth'  for i in ['D', 'P_style']]
load_path = [os.path.join(load_dir, load_filename[i]) for i in range(2)]
genre_cnt = {}

def get_file_list():
    pat = glob.glob("./datasets/test_0807/trainB/Naive*")
    dc = {}
    ls = []
    for i in pat:
        genre = i.split('-')[0].split('/')[-1].split('_')[0]
        if genre not in dc:
            dc[genre] = 1
            ls.append(i)
    return ls

def get_file_list_all(pat):
    pat = glob.glob(f"./datasets/test_0807/trainB/{pat}*")
    ls = []
    for i in pat:
        genre = i.split('-')[0].split('/')[-1].split('_')[0]
        genre_cnt[genre] += 1
        ls.append(i)
    return ls

def get_genre_list():
    pat = glob.glob("./datasets/test_0807/trainB/*")
    dc = {}
    for i in pat:
        genre = i.split('-')[0].split('/')[-1].split('_')[0]
        if genre not in dc:
            genre_cnt[genre] = 0
            dc[genre] = 1
    return list(dc.keys())

def get_directory_file_list(path):
    pat = glob.glob(path)
    return pat

def loadImgTo4DTensor(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(new_shape)
    img = to_tensor(img)
    img = img.unsqueeze(0).to("cuda:1")
    return img

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

# hyper parameter
gpu_ids = [1]
new_shape = (256, 256)
img_root_dir = "datasets/test_0807/trainB/"

# define model 
style_vgg = MSP.vgg
style_vgg.load_state_dict(torch.load('models/style_vgg.pth'))
style_vgg = nn.Sequential(*list(style_vgg.children()))
net = []
netD = MSP.StyleExtractor(style_vgg, gpu_ids)
netP_style = MSP.Projector(gpu_ids)
init_net(netD, 'normal', 0.02, gpu_ids) 
init_net(netP_style, 'normal', 0.02, gpu_ids)
net.append(netD) 
net.append(netP_style)

def get_code(img_path):
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

    for i in genre:
        image = get_file_list_all(i)

        ls = [0 for _ in range(960)]

        codes = [[] for _ in range(len(image))] 
        for k in range(len(image)):
            # image[i] = img_root_dir + image[i]
            img = loadImgTo4DTensor(image[k])
            code = net[0].forward(img,[0, 1, 2, 3])
            for j in code:
                codes[k].append(torch.flatten(j).tolist())
            cnt = 0
            for j in range(4):
                for num in range(t[j]):
                    ls[cnt] += codes[k][j][num]
                    cnt += 1
        for k in range(960):
            ls[k] /= genre_cnt[i]
        res[i] = ls

    with open("student_file.json", "w") as json_file:
        json.dump(res, json_file)

# load model
for i in range(2):
    state_dict = torch.load(load_path[i], map_location='cpu')
    net[i].load_state_dict(state_dict)

# files = get_directory_file_list("./datasets/test_0718/testB/*")
files = get_directory_file_list("./datasets/test_0807/trainB/*")

data = get_weight()

theme_list = data.keys()

ans_cnt = 0

for f in files:
    ls = get_code(f)
    theme = f.split('/')[-1].split('.')[0]
    min_ = 2e9
    gen = ""

    for key in data.keys():
        tmp = 0
        for i in range(960):
            tmp += abs(ls[i] - data[key][i])
        if(tmp < min_):
            min_ = tmp
            gen = key
    if gen in f:
        ans_cnt += 1

print(ans_cnt)

# infos = []
# sets = []
# # zero count, location
# for i in range(len(image)):
#     cnt = 0
#     info = [[] for _ in range(4)]
#     # s = set()
#     for k in range(64):
#         if codes[i][0][k] != 0:
#             cnt += 1
#             info[0].append([codes[i][0][k], k])
#             # s.add(k)
    
#     for k in range(128):
#         if codes[i][1][k] != 0:
#             cnt += 1
#             info[1].append([codes[i][1][k], k])
#             # s.add(k)

#     for k in range(256):
#         if codes[i][2][k] != 0:
#             cnt += 1
#             info[2].append([codes[i][2][k], k])
#             # s.add(k)

#     for k in range(512):
#         if codes[i][3][k] != 0:
#             cnt += 1
#             info[3].append([codes[i][3][k], k])
#             # s.add(k)

#     for i in range(4):
#         info[i].sort(reverse=True)
#     infos.append(info)
#     # sets.append(s)

# # print(infos[0][0])
# # for i in range(len(image)):
# #     genre = image[i].split('-')[0].split('/')[-1].split('_')[0]
# #     print(f"**No.{i} {genre}**")
# #     for j in range(5):
# #         print(f"Top-{j} - | {infos[i][0][j][1]:3} : {round(infos[i][0][j][0], 3):6} | {infos[i][1][j][1]:3} : {round(infos[i][1][j][0], 3):6} | {infos[i][2][j][1]:3} : {round(infos[i][2][j][0], 3):6} | {infos[i][3][j][1]:3} : {round(infos[i][3][j][0], 3):}")