import torch


def loss_get(args):
    dict_choice={'TSF':'TSF(args)._load()',
                 'OD':'OD(args)._load()'
                 }
    return eval(dict_choice[args.type])

class TSF(object):
    def __init__(self,args):
        self.args=args
    def _load(self):
        dict_loss={'mae':'torch.nn.L1Loss()',
               'mse':'torch.nn.MSELoss()'
                   }
        return eval(dict_loss[self.args.loss])

class OD(object):
    def __init__(self,args):
        self.args = args
        self.loss_confidence=torch.nn.BCELoss()
        self.loss_class = torch.nn.BCELoss()
    def _load(self):
        dict_loss={'YOLO5':'self._YOLO5'
                   }
        return eval(dict_loss[self.args.loss])
    def _iou(self,pred,true): #(batch,(Cx,Cy,w,h))
        pred_x, pred_y=pred[:,0]-1/2*pred[:,2],pred[:,1]-1/2*pred[:,3]
        true_x, true_y=true[:,0]-1/2*true[:,2],true[:,1]-1/2*true[:,3]
        x1=torch.max(pred_x,true_x)
        y1=torch.max(pred_y,true_y)
        x2=torch.min(pred_x+pred[:,2],true_x+true[:,2])
        y2=torch.min(pred_y+pred[:,3],true_y+true[:,3])
        zeros=torch.zeros(len(pred)).to(self.args.device)
        intersection=torch.max(x2-x1,zeros)*torch.max(y2-y1,zeros)
        union=(pred[:,2])*(pred[:,3])+(true[:,2])*(true[:,3])-intersection
        return intersection/union, pred_x, pred_y, true_x, true_y
    def _L1_L2(self,pred,true,pred_x, pred_y,true_x, true_y):
        x1=torch.min(pred_x,true_x)
        y1=torch.min(pred_y,true_y)
        x2=torch.max(pred_x+pred[:,2],true_x+true[:,2])
        y2=torch.max(pred_y+pred[:,3],true_y+true[:,3])
        L1=torch.square(pred[:,0]-true[:,0])+torch.square(pred[:,1]-true[:,1])
        L2=torch.square(x2-x1)+torch.square(y2-y1)
        return L1/L2
    def _ciou(self,pred,true): #(batch,(Cx,Cy,w,h))
        iou, pred_x, pred_y, true_x, true_y = self._iou(pred,true)
        L1_L2=self._L1_L2(pred,true,pred_x, pred_y,true_x, true_y)
        v=(4/(3.14159**2))*torch.square(torch.atan(true[:,2]/true[:,3])-torch.atan(pred[:,2]/pred[:,3]))
        with torch.no_grad():
            alpha=v/(1-iou+v+0.00001)
        return iou-L1_L2-alpha*v
    def _YOLO5(self,pred_batch,mask_batch,true_batch):
        loss1=0
        loss2=0
        loss3=0
        for i in range(len(self.args.OD_output[0])):
            if True in mask_batch[i]:
                pred_mask=pred_batch[i][mask_batch[i]]
                true_mask=true_batch[i][mask_batch[i]]
                pred_mask[:,0:2] = 2*pred_mask[:,0:2]-0.5
                pred_mask[:,2:4] = 4*pred_mask[:,2:4] #原版会对pred_mask[:, 2:4]做一个平方，此处没加平方，但值域一致，且线性操作不影响
                ciou=self._ciou(pred_mask[:,0:4],true_mask[:,0:4])
                loss_confidence = self.loss_confidence(pred_batch[i][...,4], true_batch[i][...,4])
                loss_class=self.loss_class(pred_mask[:,5:],true_mask[:,5:])
                loss1+=self.args.loss_param[0][i]*self.args.loss_param[i+1][0]*(1-torch.mean(ciou))
                loss2+=self.args.loss_param[0][i]*self.args.loss_param[i+1][1]*loss_confidence
                loss3+=self.args.loss_param[0][i]*self.args.loss_param[i+1][2]*loss_class
        return loss1+loss2+loss3,loss1.item(),loss2.item(),loss3.item()

if __name__ == '__main__':
    pred=torch.tensor([[10.2623, 10.0009, 5.1237, 5.0211],[30.2623, 30.0009, 5.1237, 5.0211]])
    true=torch.tensor([[30.2623, 30.0009, 5.1237, 5.0211],[30.2623, 30.0009, 5.1237, 5.0211]])
    print(OD(0)._iou(pred,true))
