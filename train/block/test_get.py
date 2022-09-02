import time
import torch
import numpy as np
import pandas as pd
from block import metric


def test_get(args,dict_dataset,model):
    dict_choice={'TSF':'TSF(args)._test(dict_dataset,model)',
                 'OD': 'OD(args)._test(dict_dataset,model)'
                 }
    return eval(dict_choice[args.type])

class TSF(object):
    def __init__(self,args):
        self.args=args
    def _test(self,dict_dataset,model):
        args=self.args
        model.eval()
        time_start = time.time()
        test_all = torch.tensor(np.stack(dict_dataset['data_test'], axis=0)).to(args.device)
        true_all = torch.tensor(np.stack(dict_dataset['true_test'], axis=0)).to(args.device)
        with torch.no_grad():
            pred_all = model(test_all)
        pred_denormalization = pred_all * torch.tensor(dict_dataset['std']).to(args.device)\
                               + torch.tensor(dict_dataset['mean']).to(args.device)
        true_denormalization = true_all * torch.tensor(dict_dataset['std']).to(args.device)\
                               + torch.tensor(dict_dataset['mean']).to(args.device)
        time_end = time.time()
        print('| 模型:{} | 数据集:{} | 变量:{} | 输入长度:{} | 预测长度:{} | 批量大小:{} | 损失函数:{} |'
              .format(args.model,args.data_name,args.TSF_column,args.TSF_input,args.TSF_output,args.batch,args.loss))
        print('| 测试集数量:{} | 测试时间:{:.2f}s | 归一化 mae:{:.4f} mse:{:.4f} | 真实 mae:{:.4f} mse:{:.4f} |'
              .format(len(true_denormalization),time_end - time_start,metric.mae(pred_all,true_all),
                      metric.mse(pred_all,true_all),metric.mae(pred_denormalization,true_denormalization),
                      metric.mse(pred_denormalization,true_denormalization)))
        if args.TSF_save:
            pred_save=pred_denormalization[:,-1].detach().cpu().numpy()
            true_save=true_denormalization[:,-1].detach().cpu().numpy()
            for i in range(len(args.TSF_column)):
                pd.DataFrame(np.stack([pred_save[:,i],true_save[:,i]],axis=1),
                             columns=[str(args.TSF_column[i])+'_pred_last',str(args.TSF_column[i])+'_true_last'])\
                    .to_csv(r'../result/'+args.name+'_'+'column'+str(args.TSF_column[i])+'_last.csv')
        if args.TSF_plot[0]:
            import matplotlib.pyplot as plt
            x=np.arange(min(args.TSF_plot[1],len(true_denormalization)))
            pred_plot = pred_denormalization[:,-1][0:len(x)].detach().cpu().numpy()
            true_plot = true_denormalization[:,-1][0:len(x)].detach().cpu().numpy()
            for i in range(len(args.TSF_column)):
                plt.title(args.name+'_column:'+str(args.TSF_column[i])+'_last')
                plt.plot(x,true_plot[:,i],color='green',label='true')
                plt.plot(x,pred_plot[:,i],color='cyan',label='pred')
                plt.savefig('../result/'+args.name+'_'+'column'+str(args.TSF_column[i])+'_last.png')
                plt.show()

class OD(object):
    def __init__(self,args):
        self.args=args
    def _test(self,dict_dataset,model):
        args = self.args
        model.eval()
        time_start = time.time()
        len_test=len(dict_dataset['img_test'])
        accuracy = torch.zeros(len_test)
        precision = torch.zeros(len_test)
        recall = torch.zeros(len_test)
        number_plot = min(len_test, args.OD_plot[1])
        pred_plot = [0 for i in range(number_plot)]
        test_plot = [0 for i in range(number_plot)]
        name_plot = [0 for i in range(number_plot)]
        index_plot=len(dict_dataset['img_test'])-number_plot
        test_loader = test_get_loader(args, dict_dataset)
        for i, (test, mask, true, name) in enumerate(test_loader):
            with torch.no_grad():
                pred = model(test)
                accuracy[i], precision[i], recall[i] = metric.acc_pre_recall(
                    args.OD_output[1], args.OD_iou_threshold, pred, mask, true)
                if args.OD_plot[0] and i >= index_plot:
                    pred_plot[i-index_plot] = pred
                    test_plot[i-index_plot] = test[0]
                    name_plot[i-index_plot] = name[0]
        accuracy = torch.mean(accuracy)
        precision = torch.mean(precision)
        recall = torch.mean(recall)
        time_end = time.time()
        print('| 模型:{} | 数据集:{} | 图片大小:{} | 类别数:{} | 损失函数:{} |'
              .format(args.model, args.data_name, args.OD_size, args.OD_class, args.loss))
        print('| 测试集数量:{} | 测试时间:{:.2f}s | 指标iou阈值:{} 准确率:{:.4f} 精确率:{:.4f} 召回率:{:.4f} |'
              .format(i+1, time_end - time_start, args.OD_iou_threshold, accuracy, precision, recall))
        if args.OD_plot[0]:
            import cv2
            anchor = torch.tensor(args.OD_anchor).to(args.device)
            for pred,test,name in zip(pred_plot,test_plot,name_plot):
                list_choose=[0 for i in range(len(args.OD_output[0]))]
                for i in range(len(args.OD_output[0])):
                    pred[i][0][:, :, :, 0]=(2*pred[i][0][:, :, :, 0]-0.5+torch.arange(args.OD_output[0][i])
                                                .to(args.device))*dict_dataset['stride'][i]
                    pred[i][0][:, :, :, 1]=(2*pred[i][0][:, :, :, 1]-0.5+torch.arange(args.OD_output[0][i])
                                                .unsqueeze(1).to(args.device))*dict_dataset['stride'][i]
                    for j in range(args.OD_output[1]):
                        pred[i][0][j, :, :, 2:4]=4*pred[i][0][j, :, :, 2:4]*anchor[i][j]
                    mask_choose = torch.where(pred[i][0][:, :, :, 4] > args.OD_plot_confidence, True, False)
                    list_choose[i]=pred[i][0][mask_choose]
                choose = torch.cat(list_choose, axis=0)
                if len(choose)==0:
                    print('!没有任何达标的预测框!')
                else:
                    choose = torch.stack(sorted(list(choose), key=lambda x: x[4], reverse=True)[:], axis=0)
                    choose[..., 0:2] = choose[..., 0:2] - 1 / 2 * choose[..., 2:4]
                    choose[..., 2:4] = choose[..., 0:2] + choose[..., 2:4]
                    img = (test.cpu().numpy() * 255).astype(np.uint8)
                    if args.OD_plot_show:
                        img_all = img.copy()
                        all_choose = choose.type(torch.int32).cpu().numpy()
                        for i in range(len(all_choose)):
                            cv2.rectangle(img_all, (all_choose[i][0],all_choose[i][1]),
                                          (all_choose[i][2],all_choose[i][3]), color=(0,255,0), thickness=2)
                    mask_choose = torch.ones(len(choose), dtype=bool)
                    index_choose = torch.arange(0, len(choose))
                    for i in range(len(choose)-1):
                        if mask_choose[i]:
                            for k, index in zip(choose[i + 1:][mask_choose[i + 1:]], index_choose[i + 1:][mask_choose[i + 1:]]):
                                if metric.iou(k[0:4], choose[i][0:4]) > args.OD_plot_threshold:
                                    mask_choose[index] = False
                    choose = choose[mask_choose].cpu().numpy()
                    _xy = choose[:, 0:4].astype(np.int32)
                    _class = np.argmax(choose[:, 5:], axis=1)
                    for i in range(len(choose)):
                        cv2.rectangle(img, (_xy[i][0], _xy[i][1]), (_xy[i][2], _xy[i][3]), color=(0, 255, 0), thickness=2)
                        cv2.putText(img, dict_dataset['class_list'][_class[i]], (_xy[i][0] + 3, _xy[i][1] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    cv2.imwrite('../result/' + args.name + '_' + name + '.jpg', img)
                    if args.OD_plot_show:
                        img = np.concatenate([img_all,img],axis=1)
                        cv2.imshow(name + '_' + str(len(all_choose)) + '_NMS_' + str(len(choose)), img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()


 # ------------------------------------------------------------------------------------------------------------------ #
def test_get_loader(args,dict_dataset):
    dict_choice={'OD': 'OD_dataset(args,dict_dataset)'
                 }
    return torch.utils.data.DataLoader(eval(dict_choice[args.type]),batch_size=1,shuffle=False,drop_last=False)

class OD_dataset(torch.utils.data.Dataset):
    def __init__(self,args,dict_dataset):
        self.args=args
        self.dict_dataset=dict_dataset
        self.anchor=torch.tensor(args.OD_anchor).to(args.device)
    def __len__(self):
        return len(self.dict_dataset['img_test'])
    def __getitem__(self, index):
        args=self.args
        dict_dataset=self.dict_dataset
        test=torch.tensor(dict_dataset['img_test'][index]/255).type(torch.float32).to(args.device)
        list_mask=[0 for i in range(len(args.OD_output[0]))]
        list_label=[0 for i in range(len(args.OD_output[0]))]
        for i,stride in enumerate(dict_dataset['stride']): #对每个输出层分别处理
            label_test = torch.tensor(dict_dataset['label_test'][index][1]).to(args.device)
            xy_stride=label_test[:,0:2]/stride #每个输出层中心点坐标比例不一样
            label_test[:, 0:2]=xy_stride%1
            label_test=label_test.repeat(args.OD_output[1],1,1)
            for j in range(args.OD_output[1]): #对每个通道分别处理宽高
                label_test[j,:,2:4]=label_test[j,:,2:4]/ self.anchor[i][j]
            mask_screen = torch.where((0.25 < label_test[:,:,2]) & (0.25 < label_test[:,:,3]) & (label_test[:,:,2] < 4) &
                               (label_test[:,:,3] < 4), True, False) #筛选出可以预测的标签
            x_grid,y_grid=xy_stride[:,0].type(torch.int32),xy_stride[:,1].type(torch.int32) #原标签
            x_add=torch.clamp(x_grid+2*torch.round(xy_stride[:,0]-x_grid).type(torch.int32)-1,0,args.OD_output[0][i]-1) #增加的标签
            y_add=torch.clamp(y_grid+2*torch.round(xy_stride[:,1]-y_grid).type(torch.int32)-1,0,args.OD_output[0][i]-1)
            label = torch.zeros((args.OD_output[1], args.OD_output[0][i], args.OD_output[0][i], 5 + args.OD_class)).to(args.device) #标签矩阵
            mask = torch.zeros((args.OD_output[1], args.OD_output[0][i], args.OD_output[0][i]), dtype=bool).to(args.device) #掩码矩阵
            for k in range(args.OD_output[1]): #对每个通道分别处理
                for j in range(len(xy_stride)): #加入原标签
                    if mask_screen[k,j]:
                        mask[k,y_grid[j],x_grid[j]]=True
                        label[k,y_grid[j],x_grid[j],:]=label_test[k,j]
            for k in range(args.OD_output[1]): #对每个通道分别处理
                for j in range(len(xy_stride)): #加入增加的标签，只有此处无标签时才可加入
                    if not mask[k,y_grid[j],x_add[j]] and mask_screen[k,j]: #加入相邻x网格
                        mask[k, y_grid[j], x_add[j]]=True
                        label[k, y_grid[j], x_add[j],:] = label_test[k,j]
                        if x_add[j]<x_grid[j]: #调整相邻x网格的x坐标
                            label[k, y_grid[j], x_add[j], 0] = label[k, y_grid[j], x_add[j], 0] + 1
                        else:
                            label[k, y_grid[j], x_add[j], 0] = label[k, y_grid[j], x_add[j], 0] - 1
                    if not mask[k,y_add[j],x_grid[j]] and mask_screen[k,j]: #加入相邻y网格
                        mask[k, y_add[j], x_grid[j]]=True
                        label[k, y_add[j], x_grid[j],:] = label_test[k,j]
                        if y_add[j] < y_grid[j]: #调整相邻y网格的y坐标
                            label[k, y_add[j], x_grid[j], 1] = label[k, y_add[j], x_grid[j], 1] + 1
                        else:
                            label[k, y_add[j], x_grid[j], 1] = label[k, y_add[j], x_grid[j], 1] - 1
            list_mask[i] = mask
            list_label[i] = label
            name = dict_dataset['label_test'][index][0]
        return test,list_mask,list_label,name
