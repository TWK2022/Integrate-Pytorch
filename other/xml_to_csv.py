#将目标检测的标签由xml格式转换为csv格式,同时将左上右下坐标转为中心点和宽高
import os
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_label=r'./label'
save_label=r'./label_save'
if not os.path.exists(save_label):
    os.makedirs(save_label)
# -------------------------------------------------------------------------------------------------------------------- #
#程序
dir_xml=os.listdir(path_label)
list_column=['class','Cx','Cy','w','h']
for i in range(len(dir_xml)):
    list_value = []
    root=ET.parse(path_label+'/'+dir_xml[i]).getroot()
    for object in root.findall('object'):
        bndbox=object.find('bndbox')
        value=[object.find('name').text,
               int(bndbox[0].text.split('.')[0]),
               int(bndbox[1].text.split('.')[0]),
               int(bndbox[2].text.split('.')[0]),
               int(bndbox[3].text.split('.')[0])
               ]
        list_value.append(value)
    df=pd.DataFrame(list_value,columns=list_column)
    w=df['w'].values-df['Cx'].values
    h=df['h'].values - df['Cy'].values
    Cx=df['Cx'].values+1/2*w
    Cy=df['Cy'].values+1/2*h
    df['Cx']=np.around(Cx).astype(np.int32)
    df['Cy']=np.around(Cy).astype(np.int32)
    df['w']=np.around(w).astype(np.int32)
    df['h'] =np.around(h).astype(np.int32)
    df.to_csv(save_label+'/'+dir_xml[i].split('.')[0]+'.csv')
print('总数:',len(dir_xml))


