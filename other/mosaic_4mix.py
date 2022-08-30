#马赛克增强，将随机4张图片拼接到一起，并填充像素(0,0,0)成为指定正方形
import os
import cv2
import numpy as np
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_img=r'./img'
path_label=r'./label'
save_img=r'./img_save'
save_label=r'./label_save'
size=640 #图片变成的形状
if not os.path.exists(save_img):
    os.makedirs(save_img)
if not os.path.exists(save_label):
    os.makedirs(save_label)
# -------------------------------------------------------------------------------------------------------------------- #
#程序
dir_img = sorted(os.listdir(path_img))
dir_label = sorted(os.listdir(path_label))
index=np.arange(len(dir_img))
np.random.shuffle(index)
size_half=size//2
def _resize(img,label):
    w0=len(img[0])
    h0=len(img)
    if w0>=h0: #宽大于高
        w=size_half
        h=int(w/w0*h0)
    else: #宽小于高
        h=size_half
        w=int(h/h0*w0)
    img=cv2.resize(img,(w,h))
    Cx = np.around(label['Cx'].values * w / w0).astype(np.int32)
    Cy = np.around(label['Cy'].values * h / h0).astype(np.int32)
    w = np.around(label['w'].values * w / w0).astype(np.int32)
    h = np.around(label['h'].values * h / h0).astype(np.int32)
    return img,Cx,Cy,w,h
for i in range(len(dir_img)//4):
    img1 = cv2.imread(path_img + '/' + dir_img[index[4 * i]])
    img2 = cv2.imread(path_img + '/' + dir_img[index[4 * i + 1]])
    img3 = cv2.imread(path_img + '/' + dir_img[index[4 * i + 2]])
    img4 = cv2.imread(path_img + '/' + dir_img[index[4 * i + 3]])
    label1 = pd.read_csv(path_label + '/' + dir_label[index[4 * i]])
    label2 = pd.read_csv(path_label + '/' + dir_label[index[4 * i + 1]])
    label3 = pd.read_csv(path_label + '/' + dir_label[index[4 * i + 2]])
    label4 = pd.read_csv(path_label + '/' + dir_label[index[4 * i + 3]])
    img1, Cx1, Cy1, w1, h1 = _resize(img1, label1)
    img2, Cx2, Cy2, w2, h2 = _resize(img2, label2)
    img3, Cx3, Cy3, w3, h3 = _resize(img3, label3)
    img4, Cx4, Cy4, w4, h4 = _resize(img4, label4)
    Cx1 += size_half - len(img1[0])
    Cx2 += size_half
    Cx3 += size_half - len(img3[0])
    Cx4 += size_half
    Cy1 += size_half - len(img1)
    Cy2 += size_half - len(img2)
    Cy3 += size_half
    Cy4 += size_half
    img1 = cv2.copyMakeBorder(img1, size_half - len(img1), 0, size_half - len(img1[0]), 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img2 = cv2.copyMakeBorder(img2, size_half - len(img2), 0, 0, size_half - len(img2[0]), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img3 = cv2.copyMakeBorder(img3, 0, size_half - len(img3), size_half - len(img3[0]), 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img4 = cv2.copyMakeBorder(img4, 0, size_half - len(img4), 0, size_half - len(img4[0]), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img=np.concatenate([np.concatenate([img1,img2],axis=1),np.concatenate([img3,img4],axis=1)],axis=0)
    label = pd.DataFrame(columns=['class','Cx','Cy','w','h'])
    label['class'] = np.concatenate([label1['class'].values, label2['class'].values,
                                     label3['class'].values, label4['class'].values], axis=0)
    label['Cx']=np.concatenate([Cx1,Cx2,Cx3,Cx4],axis=0)
    label['Cy']=np.concatenate([Cy1,Cy2,Cy3,Cy4],axis=0)
    label['w']=np.concatenate([w1,w2,w3,w4],axis=0)
    label['h']=np.concatenate([h1,h2,h3,h4],axis=0)
    name=dir_img[index[i]].split('.')[0]+'_masaic_4mix.'+dir_img[index[i]].split('.')[1]
    cv2.imwrite(save_img + '/' + name,img)
    label.to_csv(save_label+'/'+ name)
print('总数:',i+1)

#检验
class_=label['class']
frame=label[['Cx','Cy','w','h']].values
frame[:,0:2] = frame[:,0:2] - 1/2*frame[:,2:4]
frame[:, 2:4] = frame[:, 2:4] + frame[:, 0:2]
frame=frame.astype(np.int32)
for j in range(len(frame)):
    cv2.rectangle(img, (frame[j][0], frame[j][1]), (frame[j][2], frame[j][3]), color=(0, 255, 0), thickness=2)
    cv2.putText(img, class_[j], (frame[j][0]+3, frame[j][1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
cv2.imshow(name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
