from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from PIL import Image

df = pd.read_csv("weights.csv")

train_dt = df.iloc[:, 2:]
model = KMeans(n_clusters=30, algorithm='auto', random_state=42)
predict = pd.DataFrame(model.fit_predict(train_dt))

min_dist = np.min(cdist(train_dt, model.cluster_centers_, 'euclidean'), axis = 1)
# Y = pd.DataFrame(min_dist, index=train_dt.index, columns=['Center_euclidean_dist'])
# Z = pd.DataFrame(predict, index=train_dt.index, columns=['cluster_ID'])
Y = pd.DataFrame(min_dist, columns=['Center_euclidean_dist'])
Y["label"] = predict

Y2 = Y.groupby('label')
print(Y[Y['label'] == 2].sort_values('Center_euclidean_dist')[:2].index)
for i in range(30):
    for idx in Y[Y['label'] == i].sort_values('Center_euclidean_dist')[:5].index:
        fname = df.iloc[idx]['file_path']
        # print(fname)``
        img = Image.open(fname)
        fname = f"./centroids/{i}_{fname.split('/')[-1]}"
        img.save(fname)

# grouped = PAP.groupby(['cluster_ID'])
# print(grouped.idxmin())
# print(grouped)

idx = []

# for cluster in model.cluster_centers_:
#     for line in range(df.shape[0]):
#         flag = True
#         for i in range(960):
#             if round(cluster[i],3) != round(train_dt.iloc[line, i],3):
#                 flag = False
#                 break
#         if flag:
#             print(1)
#             break

# df.to_csv("./k30.csv")
# 
# print(predict)