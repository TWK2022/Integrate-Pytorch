#将seq格式中的图片取出并转换为jpg格式
import os.path
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_img=r'./img'
save_img=r'./img_save'
if not os.path.exists(save_img):
    os.makedirs(save_img)
# -------------------------------------------------------------------------------------------------------------------- #
#程序
dir_seq = os.listdir(path_img)
for i in range(len(dir_seq)):
    with open(path_img+'/'+dir_seq[i], 'rb+') as f:
        string = f.read().decode('latin-1')
        splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
        strlist = string.split(splitstring)
    save_file=save_img+'/'+dir_seq[i].split('.')[0]
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    count = 0
    for img in strlist:
        if count > 0:
            i = open(save_file+'/'+str(count) + '.jpg', 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1
