import pandas as pd
import numpy as np
import utils
data = pd.read_csv("./weights.csv")
data = data.drop(['Unnamed: 0'], axis=1)

# 작가 목록
author_list = utils.get_author_list() 


author = []
for i in range(data.shape[0]):
    curAuthor = data.iloc[i,0].split('/')[-1].split('_')[-2]
    author.append(curAuthor)

df = pd.DataFrame(columns = ["file_path", *range(960)])

data['theme'] = author

dc2 = { i : 0 for i in range(960) }

for g in author_list:
    cond = data['theme'] == g
    d = data[cond].var().to_dict()
    e = data[cond].mean().to_dict()

    dc = {}
    for i in range(960):
        dc[i] = [d[str(i)], e[str(i)]]
        
    # 분산값으로 오름차순
    sd = sorted(dc.items(), key = lambda item: item[1][0])
    # # 평균 값으로 오름차순
    # sd = sorted(dc.items(), key = lambda item: -item[1][1])
    # print(dc)
    # print("**theme** : ", g)
    # print("variance and expectation sort by Variance : \n", end = '')

    cnt = 0
    for j in sd:
        if j[1][0] == 0.0 and j[1][0] != np.nan:
            continue
        else:
            dc2[j[0]] += (10-cnt)
            cnt += 1
        if cnt == 10:
            break

sd2 = sorted(dc2.items(), key = lambda item: item[1])
print(sd2)
    # print("variance and expectation sort by Expectation : \n", sd[-50:])