#在图片上画出框，标签为CSV格式，包含CxCywh
import os
import cv2
import numpy as np
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_img=r'./img'
path_label=r'./label'
# -------------------------------------------------------------------------------------------------------------------- #
#程序
dir_img = sorted(os.listdir(path_img))
dir_label = sorted(os.listdir(path_label))
for i in range(len(dir_img)):
    img=cv2.imread(path_img+'/'+dir_img[i])
    label=pd.read_csv(path_label+'/'+dir_label[i])
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
print('总数:',len(dir_img))
