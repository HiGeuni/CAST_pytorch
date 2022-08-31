import os
import glob

pat = glob.glob("./datasets/test_0807/trainB/*")
pat2 = glob.glob("/home/khyo/wikiart/wikiart/*")

category_list = []

dc = {}
# for i in range(len(pat2)):
#     if os.path.isdir(pat2[i]):
#         category = pat2[i].split('/')[-1]
#         category_list.append(category)
#         dc[category] = 0

# print(dc)
ls = []

for i in pat:
    genre = i.split('-')[0].split('/')[-1].split('_')[0]
    if genre not in dc:
        dc[genre] = 1
        ls.append(i)

ls.sort()
for i in ls:
    print(i)
print(len(ls))