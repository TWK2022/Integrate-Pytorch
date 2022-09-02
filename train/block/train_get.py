import time
import torch
from block import metric

def train_get(args,dict_dataset,model,loss):
    dict_choice={'TSF':'TSF(args)._train(dict_dataset,model,loss)',
             'OD': 'OD(args)._train(dict_dataset,model,loss)'
                }
    return eval(dict_choice[args.type])

class TSF(object):
    def __init__(self,args):
        self.args=args
    def _train(self,dict_dataset,model,loss):
        args=self.args
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epoch + 1):
            time_start = time.time()
            print('| 第{}轮训练开始 | 训练集数量:{} | 批量大小:{} |'.format(epoch, len(dict_dataset['data_train']), args.batch))
            train_loader=train_get_loader(args,dict_dataset)
            for item,(train_batch,true_batch) in enumerate(train_loader):
                pred_batch = model(train_batch.to(args.device))
                loss_batch = loss(pred_batch,true_batch.to(args.device))
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                if (item+1) % args.train_show == 0:
                    print('| {} | 迭代次数:{} | 本批量loss({}):{:.4f} |'.format(args.model, item+1, args.loss, loss_batch))
            time_end = time.time()
            print('| {} | 本轮训练结束 时间:{:.2f}s |'.format(args.model,time_end - time_start))
        return model

class OD(object):
    def __init__(self,args):
        self.args=args
    def _train(self,dict_dataset,model,loss):
        args=self.args
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epoch + 1):
            time_start = time.time()
            print('| 第{}轮训练开始 | 训练集数量:{} | 批量大小:{} |'.format(epoch,len(dict_dataset['img_train']),args.batch))
            train_loader=train_get_loader(args,dict_dataset)
            for item,(train_batch,mask_batch,true_batch) in enumerate(train_loader):
                pred_batch = model(train_batch)
                loss_batch,loss1_item,loss2_item,loss3_item = loss(pred_batch,mask_batch,true_batch)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                if args.train_show[0]:
                    if (item+1) % args.train_show[1] == 0:
                        accuracy, precision, recall = metric.acc_pre_recall(
                            args.OD_output[1],args.OD_iou_threshold,pred_batch,mask_batch,true_batch)
                        print('| {} | 迭代次数:{} | 本批量loss({}):边框_{:.4f} 置信度_{:.4f} 分类_{:.4f} '
                              '| 指标iou阈值:{} 准确度:{:.4f} 精确度:{:.4f} 召回率:{:.4f} |'
                              .format(args.model, item + 1, args.loss, loss1_item, loss2_item, loss3_item,
                                      args.OD_iou_threshold, accuracy, precision, recall))
            accuracy, precision, recall = metric.acc_pre_recall(
                args.OD_output[1], args.OD_iou_threshold, pred_batch, mask_batch, true_batch)
            print('| {} | 最后一次迭代:{} | 本批量loss({}):边框_{:.4f} 置信度_{:.4f} 分类_{:.4f} '
                  '| 指标iou阈值:{} 准确度:{:.4f} 精确度:{:.4f} 召回率:{:.4f} |'
                  .format(args.model, item + 1, args.loss, loss1_item, loss2_item, loss3_item,
                          args.OD_iou_threshold, accuracy, precision, recall))
            time_end = time.time()
            print('| {} | 本轮训练结束 时间:{:.2f}s |'.format(args.model,time_end - time_start))
        return model

 # ------------------------------------------------------------------------------------------------------------------ #
def train_get_loader(args,dict_dataset):
    dict_choice={'TSF':'TSF_dataset(args,dict_dataset)',
                 'OD': 'OD_dataset(args,dict_dataset)'
                 }
    return torch.utils.data.DataLoader(eval(dict_choice[args.type]),
                                       batch_size=args.batch,shuffle=True,drop_last=True)

class TSF_dataset(torch.utils.data.Dataset):
    def __init__(self,args,dict_dataset):
        self.args=args
        self.dict_dataset=dict_dataset
    def __len__(self):
        return len(self.dict_dataset['data_train'])
    def __getitem__(self, index):
        train=self.dict_dataset['data_train'][index]
        true =self.dict_dataset['true_train'][index]
        return train,true

class OD_dataset(torch.utils.data.Dataset):
    def __init__(self,args,dict_dataset):
        self.args=args
        self.dict_dataset=dict_dataset
        self.anchor=torch.tensor(args.OD_anchor).to(args.device)
    def __len__(self):
        return len(self.dict_dataset['img_train'])
    def __getitem__(self, index):
        args=self.args
        dict_dataset=self.dict_dataset
        train=torch.tensor(dict_dataset['img_train'][index]/255).type(torch.float32).to(args.device)
        list_mask=[0 for i in range(len(args.OD_output[0]))]
        list_label=[0 for i in range(len(args.OD_output[0]))]
        for i,stride in enumerate(dict_dataset['stride']): #对每个输出层分别处理
            label_train = torch.tensor(dict_dataset['label_train'][index][1]).to(args.device)
            xy_stride=label_train[:,0:2]/stride #每个输出层中心点坐标比例不一样
            label_train[:, 0:2]=xy_stride%1
            label_train=label_train.repeat(args.OD_output[1],1,1)
            for j in range(args.OD_output[1]): #对每个通道分别处理宽高
                label_train[j,:,2:4]=label_train[j,:,2:4]/ self.anchor[i][j]
            mask_screen = torch.where((0.25 < label_train[:,:,2]) & (0.25 < label_train[:,:,3]) & (label_train[:,:,2] < 4) &
                               (label_train[:,:,3] < 4), True, False) #筛选出可以预测的标签
            x_grid,y_grid=xy_stride[:,0].type(torch.int32),xy_stride[:,1].type(torch.int32) #原标签
            x_add=torch.clamp(x_grid+2*torch.round(xy_stride[:,0]-x_grid).type(torch.int32)-1,0,args.OD_output[0][i]-1) #增加的标签
            y_add=torch.clamp(y_grid+2*torch.round(xy_stride[:,1]-y_grid).type(torch.int32)-1,0,args.OD_output[0][i]-1)
            label = torch.zeros((args.OD_output[1], args.OD_output[0][i], args.OD_output[0][i], 5 + args.OD_class)).to(args.device) #标签矩阵
            mask = torch.zeros((args.OD_output[1], args.OD_output[0][i], args.OD_output[0][i]), dtype=bool).to(args.device) #掩码矩阵
            for k in range(args.OD_output[1]): #对每个通道分别处理
                for j in range(len(xy_stride)): #加入原标签
                    if mask_screen[k,j]:
                        mask[k,y_grid[j],x_grid[j]]=True
                        label[k,y_grid[j],x_grid[j],:]=label_train[k,j]
            for k in range(args.OD_output[1]): #对每个通道分别处理
                for j in range(len(xy_stride)): #加入增加的标签，只有此处无标签时才可加入
                    if not mask[k,y_grid[j],x_add[j]] and mask_screen[k,j]: #加入相邻x网格
                        mask[k, y_grid[j], x_add[j]]=True
                        label[k, y_grid[j], x_add[j],:] = label_train[k,j]
                        if x_add[j]<x_grid[j]: #调整相邻x网格的x坐标
                            label[k, y_grid[j], x_add[j], 0] = label[k, y_grid[j], x_add[j], 0] + 1
                        else:
                            label[k, y_grid[j], x_add[j], 0] = label[k, y_grid[j], x_add[j], 0] - 1
                    if not mask[k,y_add[j],x_grid[j]] and mask_screen[k,j]: #加入相邻y网格
                        mask[k, y_add[j], x_grid[j]]=True
                        label[k, y_add[j], x_grid[j],:] = label_train[k,j]
                        if y_add[j] < y_grid[j]: #调整相邻y网格的y坐标
                            label[k, y_add[j], x_grid[j], 1] = label[k, y_add[j], x_grid[j], 1] + 1
                        else:
                            label[k, y_add[j], x_grid[j], 1] = label[k, y_add[j], x_grid[j], 1] - 1
            list_mask[i] = mask
            list_label[i] = label
        return train,list_mask,list_label