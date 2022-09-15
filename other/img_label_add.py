#对图片进行HSV中H通道色彩变换、水平翻转和加高斯噪声
import os
import cv2
import numpy as np
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_img= r'./img'
path_label= r'./label'
save_img=r'./img_save'
save_label=r'./label_save'
HSV=True
flip=True
noise=True
if not os.path.exists(save_img):
    os.makedirs(save_img)
if not os.path.exists(save_label):
    os.makedirs(save_label)
# -------------------------------------------------------------------------------------------------------------------- #
#程序
dir_img = sorted(os.listdir(path_img))
dir_label = sorted(os.listdir(path_label))
for i in range(len(dir_img)):
    name, format = dir_img[i].split('.')
    img = cv2.imread(path_img + '/' + dir_img[i])
    label = pd.read_csv(path_label + '/' + dir_label[i])
    frame=label[['Cx','Cy','w','h']].values
    if HSV:
        name += '_H'
        H_add = np.random.randint(0, 10, 1)[0]  # H通道增加量
        img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img[:,:,0]+=H_add
        img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    if flip:
        name += '_flip'
        w=img.shape[1]
        img=cv2.flip(img,1)
        frame[:,0]=w-frame[:,0]
    if noise:
        img = img / 255 + np.random.normal(0, 0.04, img.shape)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    label[['Cx', 'Cy', 'w', 'h']]=frame
    label.to_csv(save_label+'/'+name+'.csv',index=False)
    cv2.imwrite(save_img+'/'+name+'.'+format,img)
print('总数:',len(dir_img))

# 检验
x_min=(frame[:,0]-1/2*frame[:,2]).astype(np.int32)
y_min=(frame[:,1]-1/2*frame[:,3]).astype(np.int32)
x_max=(frame[:,0]+1/2*frame[:,2]).astype(np.int32)
y_max=(frame[:,1]+1/2*frame[:,3]).astype(np.int32)
for i in range(len(frame)):
    cv2.rectangle(img, (x_min[i], y_min[i]), (x_max[i], y_max[i]), color=(0, 255, 0), thickness=2)
cv2.imshow('检验', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
