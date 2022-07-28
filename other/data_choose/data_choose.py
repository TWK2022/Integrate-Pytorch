#根据指定条件筛选图片和标签
import os
import cv2
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path=r'.\all.csv'
path_img=r'.\img'
path_label=r'.\label'
save_img=r'.\img_choose'
save_label=r'.\label_choose'
# -------------------------------------------------------------------------------------------------------------------- #
#程序
df = pd.read_csv(path)
dir_img = os.listdir(path_img)
dir_label = os.listdir(path_label)
number=df['number'].values
count=0
for i in range(len(number)):
    if number[i]<80:
        cv2.imwrite(save_img+'/'+dir_img[i],cv2.imread(path_img+'/'+dir_img[i]))
        pd.read_csv(path_label+'/'+dir_label[i]).to_csv(save_label+'/'+dir_label[i])
        count+=1
print('总数:',count)