import torch

def mae(pred,true):
    return torch.mean(abs(pred-true))

def mse(pred,true):
    return torch.mean(torch.square(pred-true))

def acc_pre_recall(layer,threshold,pred,mask,true):
    P=0
    N=0
    TP=0
    TN=0
    for i in range(layer):
        if True in mask[i]:
            pred_mask=pred[i][mask[i]]
            true_mask=true[i][mask[i]]
            mask_back=(mask[i]==False)
            pred_confidence = pred_mask[:, 4]
            pred_confidence_back=pred[i][mask_back][:,4]
            pred_class=torch.argmax(pred_mask[:,5:],axis=1)
            true_class=torch.argmax(pred_mask[:,5:],axis=1)
            pred_mask[:, 0:2] = 2 * pred_mask[:, 0:2] - 0.5
            pred_mask[:, 2:4] = 4 * pred_mask[:, 2:4]
            pred_mask[..., 0:2] = pred_mask[..., 0:2] - 1 / 2 * pred_mask[..., 2:4]
            pred_mask[..., 2:4] = pred_mask[..., 0:2] + pred_mask[..., 2:4]
            true_mask[..., 0:2] = true_mask[..., 0:2] - 1 / 2 * true_mask[..., 2:4]
            true_mask[..., 2:4] = true_mask[..., 0:2] + true_mask[..., 2:4]
            mask_TP=torch.where((pred_confidence>=0.5)&(pred_class==true_class)&
                                      (iou(pred_mask[...,0:4],true_mask[...,0:4])>threshold),True,False)
            mask_TN=torch.where(pred_confidence_back<0.5,True,False)
            P += len(pred_confidence)
            N += len(pred_confidence_back)
            TP += len(pred_confidence[mask_TP])
            TN += len(pred_confidence_back[mask_TN])
    return (TP+TN)/(P+N), TP/(TP+N-TN), TP/P

def iou(pred,true): #(x,y,w,h)
    if len(pred.shape)==1:
        x1=torch.max(pred[0],true[0])
        y1=torch.max(pred[1],true[1])
        x2=torch.min(pred[2],true[2])
        y2=torch.min(pred[3],true[3])
        intersection=max(x2-x1,0)*max(y2-y1,0)
        union=(pred[2]-pred[0])*(pred[3]-pred[1])+(true[2]-true[0])*(true[3]-true[1])-intersection
    else:
        x1=torch.max(pred[:,0],true[:,0])
        y1=torch.max(pred[:,1],true[:,1])
        x2=torch.min(pred[:,2],true[:,2])
        y2=torch.min(pred[:,3],true[:,3])
        zeros = torch.zeros(len(pred)).to(pred.device.type)
        intersection=torch.max(x2-x1,zeros)*torch.max(y2-y1,zeros)
        union=(pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])+(true[:,2]-true[:,0])*(true[:,3]-true[:,1])-intersection
    return intersection/union

if __name__ == '__main__':
    pred=torch.tensor([[10., 10., 20., 20.],[10., 10., 20., 20.]])
    true=torch.tensor([[10., 10., 20., 20.],[10., 10., 20., 20.]])
    print(iou(pred,true))
