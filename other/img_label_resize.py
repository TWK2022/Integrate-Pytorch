#将图片等比缩放为指定形状，并填充像素(0,0,0)成为正方形
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
for i in range(len(dir_img)):
    img=cv2.imread(path_img+'/'+dir_img[i])
    label=pd.read_csv(path_label+'/'+dir_label[i])
    w0=len(img[0])
    h0=len(img)
    if w0==h0==size:
        cv2.imwrite(save_img + '/' + dir_img[i], img)
        label.to_csv(save_label + '/' + dir_label[i])
        continue
    if w0>=h0: #宽大于高
        w=size
        h=int(len(img)*size/len(img[0]))
        img=cv2.resize(img,(w,h))
        add_y=(w-h)//2
        img=cv2.copyMakeBorder(img,add_y,w-h-add_y,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        Cx=label['Cx'].values*w/w0
        Cy = label['Cy'].values * h / h0+add_y
        w=label['w'].values * w / w0
        h=label['h'].values * h / h0
    else: #宽小于高
        w=int(len(img[0])*size/len(img))
        h=size
        img=cv2.resize(img,(w,h))
        add_x=(h-w)//2
        img=cv2.copyMakeBorder(img,0,0,add_x,h-w-add_x,cv2.BORDER_CONSTANT,value=(0,0,0))
        Cx=label['Cx'].values*w/w0+add_x
        Cy = label['Cy'].values * h / h0
        w=label['w'].values * w / w0
        h=label['h'].values * h / h0
    label=label.drop(['Cx','Cy','w','h'],axis=1)
    label['Cx']=np.around(Cx).astype(np.int32)
    label['Cy']=np.around(Cy).astype(np.int32)
    label['w']=np.around(w).astype(np.int32)
    label['h']=np.around(h).astype(np.int32)
    cv2.imwrite(save_img + '/' + dir_img[i],img.astype(np.uint8))
    label.to_csv(save_label+'/'+dir_label[i])
print('总数:',len(dir_img))

#检验
class_=label['class']
frame=label[['Cx','Cy','w','h']].values
frame[:,0:2] = frame[:,0:2] - 1/2*frame[:,2:4]
frame[:, 2:4] = frame[:, 2:4] + frame[:, 0:2]
frame=frame.astype(np.int32)
for j in range(len(frame)):
    cv2.rectangle(img, (frame[j][0], frame[j][1]), (frame[j][2], frame[j][3]), color=(0, 255, 0), thickness=2)
    cv2.putText(img, class_[j], (frame[j][0]+3, frame[j][1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
cv2.imshow(dir_img[i], img)
cv2.waitKey(0)
cv2.destroyAllWindows()
