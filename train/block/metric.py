import torch
import collections

def mae(pred,true):
    return torch.mean(abs(pred-true))

def mse(pred,true):
    return torch.mean(torch.square(pred-true))

def accuracy_precision(layer,threshold,pred,mask,true):
    correct_accuracy=0
    all_accuracy=0
    correct_precision=0
    all_precision=0
    for i in range(layer):
        if True in mask[i]:
            pred_confidence=pred[i][mask[i]][:,4]
            pred_class=torch.argmax(pred[i][mask[i]][:,5:],axis=1)
            true_class=torch.argmax(true[i][mask[i]][:,5:],axis=1)
            mask_back=(mask[i]==False)
            pred_confidence_back=pred[i][mask_back][:,4]
            mask_correct1=torch.where((pred_confidence>threshold)&(pred_class==true_class),True,False)
            mask_correct2=torch.where(pred_confidence_back<threshold,True,False)
            correct_accuracy+=len(pred_confidence[mask_correct1])+len(pred_confidence_back[mask_correct2])
            all_accuracy+=len(pred_confidence)+len(pred_confidence_back)
            correct_precision += len(pred_confidence[mask_correct1])
            all_precision += len(pred_confidence)
    return correct_accuracy/all_accuracy,correct_precision/all_precision

def iou(pred,true): #(x1,y1,x2,y2)
    x1=torch.max(pred[0],true[0])
    y1=torch.max(pred[1],true[1])
    x2=torch.min(pred[2],true[2])
    y2=torch.min(pred[3],true[3])
    intersection=max(x2-x1,0)*max(y2-y1,0)
    union=(pred[2]-pred[0])*(pred[3]-pred[1])+(true[2]-true[0])*(true[3]-true[1])-intersection
    return intersection/union

if __name__ == '__main__':
    pred=torch.tensor([543.2623, 343.0009, 577.1237, 373.0211])
    true=torch.tensor([543.2623, 343.0009, 577.1237, 373.0211])
    print(iou(pred,true))
