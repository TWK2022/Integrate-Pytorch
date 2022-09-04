#SCINet: model_param #[奇偶分解层数，模型堆叠数] 推荐[3,1] #loss_param 无

#LSTM/LSTM_CNN: model_param #[LSTM输出维度,LSTM层数] 推荐[64,2] #loss_param 无

#YOLO5: model_param #[第一层输出维度,C3层残差数] 推荐YOLO5n[16,1],YOLO5s[32,1],YOLO5m[48,2],YOLO5l[64,3],YOLO5x[80,4]
#YOLO5: loss_param #[每层输出权重,3*[边框权重,置信度权重,类别权重]] 推荐[[1/3,1/3,1/3],[0.2,0.5,0.3],[0.3,0.4,0.3],[0.4,0.4,0.3]]

import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.loss_get import loss_get
from block.train_get import train_get
from block.save_get import save_get
from block.test_get import test_get

# 参数设置
parser = argparse.ArgumentParser(description='!!!')
parser.add_argument('--type', default='OD', type=str,choices=['TSF','OD'], help='|任务类型|')
parser.add_argument('--data_divide', default=[8,2], type=list, help='|训练集:测试集的比例划分:[a,b]|')
parser.add_argument('--data_root', default='./dataset/', type=str,choices=['./dataset/'],help='|数据根目录路径|')
parser.add_argument('--data_name', default='lamp', type=str, help='|数据文件名。TSF只需要一个csv文件，变量在TSF_column中指定;'
                    'OD还需要两个子文件夹img和label，label中标签为csv文件，需有列名[class,Cx,Cy,w,h]|')

parser.add_argument('--model', default='YOLO5', type=str, choices=['LSTM','LSTM_CNN','SCINet','YOLO5'],help='|模型选择|')
parser.add_argument('--model_param', default=[16,1], type=list,help='|模型参数:[1,2,...]|')
parser.add_argument('--model_save', default=[True,'pkl'], type=list, help='|训练完后是否保存模型和保存的类型|')
parser.add_argument('--model_test', default=True, type=bool,choices=[True,False],help='|不训练直接测试模型，同时不保存模型|')
parser.add_argument('--model_continue', default=True, type=str,choices=[True,False],help='|是否接着训练已有模型|')

parser.add_argument('--epoch', default=10, type=int,help='|训练轮数|')
parser.add_argument('--batch', default=16, type=int,help='|训练批量大小|')
parser.add_argument('--loss', default='YOLO5', type=str, choices=['mae','mse','YOLO5'],help='|损失函数|')
parser.add_argument('--loss_param', default=[[1/3,1/3,1/3],[0.2,0.6,0.2],[0.3,0.5,0.2],[0.4,0.4,0.2]], type=list, help='|损失权重|')
parser.add_argument('--lr', default=0.0002, type=int,help='|初始学习率，训练中采用adam算法，多次训练调整时lr应调为0.0002以下|')
parser.add_argument('--device', default='cuda', type=str,choices=['cuda','cpu'],help='|训练设备|')
parser.add_argument('--train_show', default=[True, 10], type=list, help='|训练时多少次迭代计算一次指标，无论是否开启每轮最后一次迭代都会计算一次指标|')

parser.add_argument('--TSF_column', default=[1,3], type=list, help='|TSF选择变量所在的列[0,1,2,...]|')
parser.add_argument('--TSF_input', default=64, type=int, help='|TSF输入训练长度:4*n|')
parser.add_argument('--TSF_output', default=32, type=int, help='|TSF输出预测长度:4*n|')
parser.add_argument('--TSF_save', default=True, type=bool, choices=[True,False],help='|TSF保存各变量的最后一个预测值与真实值为csv|')
parser.add_argument('--TSF_plot', default=[True,300], type=list,help='|TSF是否画出每种变量的最后一个预测值组成的图像及画出的数据量|')

parser.add_argument('--OD_size', default=640, type=int, help='|OD输入图片大小|')
parser.add_argument('--OD_class', default=1, type=int, help='|OD类别数|')
parser.add_argument('--OD_smooth', default=[0.05,0.95], type=list, help='|OD标签平滑的值|')
parser.add_argument('--OD_output', default=[[80,40,20],3], type=list, help='|OD输出形状|')
parser.add_argument('--OD_anchor', default=[[[10,13],[16,30],[33,23]],[[30,61],[62,45],[59,119]],[[116,90],[156,198],[373,326]]],type=list, help='|OD先验框|')
parser.add_argument('--OD_screen_confidence', default=0.8, type=float, help='|OD非极大值抑制前先根据置信度筛选框|')
parser.add_argument('--OD_screen_iou', default=0.5, type=float, help='|OD指标计算的iou筛选阈值，使用的指标计算在nms前|')
parser.add_argument('--OD_nms_iou', default=0.2, type=float, help='|OD画图时消除同类别重叠框的iou阈值，越低筛选出的框越少|')
parser.add_argument('--OD_plot', default=[True,2], type=list, help='|OD是否保存测试的图片和保存图片数量(倒数几张)|')
parser.add_argument('--OD_plot_show', default=True, type=list, choices=[True,False], help='|OD是否在页面显示非极大值抑制前后的图片对比|')


args = parser.parse_args()
args.name=args.type+'_'+args.model+'_'+args.data_name
print('| args:{} |'.format(args))

# 训练设置
torch.manual_seed(999) #为CPU设置随机种子
torch.cuda.manual_seed_all(999) #为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True #固定每次返回的卷积算法
torch.backends.cudnn.enabled = True #cuDNN使用非确定性算法
torch.backends.cudnn.benchmark = False #在训练前cuDNN会先搜寻每个卷积层最合适实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False

# 训练准备
dict_dataset=data_get(args)
model=model_get(args).to(args.device)
loss=loss_get(args)
print('| 模型:{} | 损失函数:{} |'.format(args.model,args.loss))

# 开始训练
if not args.model_test:
    model=train_get(args,dict_dataset,model,loss)
    
# 模型保存
if not args.model_test:
    save_get(args,model)

# 测试
test_get(args,dict_dataset,model)
