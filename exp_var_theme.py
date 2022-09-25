import pandas as pd
import utils
data = pd.read_csv("./weights.csv")
data = data.drop(['Unnamed: 0'], axis=1)

print(data.iloc[4])
genre_list = utils.get_theme_list() 

genre = []
for i in range(data.shape[0]):
    curTheme = data.iloc[i,0].split('/')[-1].split('_')[0]
    genre.append(curTheme)

data['theme'] = genre

dc2 = { i : 0 for i in range(960) }

for g in genre_list:
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
    print("**theme** : ", g)
    print("variance and expectation sort by Variance : \n", end = '')

    cnt = 0
    for j in sd:
        if j[1][0] <= 20.0:
            continue
        else:
            dc2[j[0]] += (5-cnt)
            cnt += 1
            print(j)
        if cnt == 5:
            break

sd2 = sorted(dc2.items(), key = lambda item: item[1])
print(sd2)
    # print("variance and expectation sort by Expectation : \n", sd[-50:])