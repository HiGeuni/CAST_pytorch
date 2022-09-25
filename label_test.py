import pandas as pd
import utils

def f(x):
    return dc[x]

lengthOfK = 100

df = pd.read_csv("k100.csv")

theme_list = utils.get_theme_list()

dc = [{} for _ in range(lengthOfK)]
for i in range(lengthOfK):
    for theme in theme_list:
        dc[i][theme] = 0

for i in range(df.shape[0]):
    curTheme = df.iloc[i,1].split('/')[-1].split('_')[0]
    dc[df.iloc[i, -1]][curTheme] += 1
    
res = {}

for i in range(lengthOfK):
    key_max = max(dc[i].keys(), key=(lambda k:dc[i][k]))
    if key_max not in res:
        res[key_max] = 0
    res[key_max] += 1

for key in res:
    print(key, res[key])


print(len(theme_list))