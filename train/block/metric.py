import torch

def mae(pred,true):
    return torch.mean(abs(pred-true))

def mse(pred,true):
    return torch.mean(torch.square(pred-true))

def nmsbefore_acc_pre_recall(pred,mask,true,screen_confidence,screen_iou,key=False):  # 对网络输出(单个/批量)求非极大值抑制前的指标
    tp_fn=0
    tn_fp=0
    tp=0
    tn=0
    for i in range(len(pred)):
        if True in mask[i]:
            pred_mask = pred[i][mask[i]]
            true_mask = true[i][mask[i]]
            pred_mask[..., 0:2] = 2 * pred_mask[..., 0:2] - 0.5  # 将输出坐标与标签对应
            pred_mask[..., 2:4] = 4 * pred_mask[..., 2:4]
            mask_back = (mask[i] == False)
            pred_confidence = pred_mask[..., 4]
            pred_confidence_back = pred[i][mask_back][..., 4]
            pred_class = torch.argmax(pred_mask[..., 5:], axis=1)
            true_class = torch.argmax(pred_mask[..., 5:], axis=1)
            mask_TP = torch.where((pred_confidence >= screen_confidence) & (pred_class == true_class) &
                                  (iou(pred_mask[..., 0:4], true_mask[..., 0:4], True) > screen_iou), True, False)
            mask_TN = torch.where(pred_confidence_back < 0.5, True, False)
            tp_fn += len(pred_confidence)
            tn_fp += len(pred_confidence_back)
            tp += len(pred_confidence[mask_TP])
            tn += len(pred_confidence_back[mask_TN])
    if key:
        return tp_fn, tn_fp, tp, tn
    return (tp+tn)/(tp_fn+tn_fp), tp/(tp+tn_fp-tn), tp/tp_fn


def nms(choose, class_number, nms_iou):  # 会对所有类别分别进行非极大值抑制与合并
    if len(choose.shape) == 1:
        return choose
    choose = torch.stack(sorted(list(choose), key=lambda x: x[4], reverse=True), axis=0)  # 按置信度排序
    class_ = torch.argmax(choose[:, 5: 5+class_number], axis=1)
    choose_nms = []
    for i in range(class_number):
        mask = torch.where(i == class_, True, False)
        list_choose = list(choose[mask])
        j = 1
        while len(list_choose) > j:
            a = list_choose[j-1]
            k = j
            while len(list_choose) > k:
                if iou(a, list_choose[k]) > nms_iou:
                    list_choose.pop(k)
                else:
                    k += 1
            j += 1
        choose_nms.extend(list_choose)
    return torch.stack(choose_nms,axis=0)

def iou(pred,true,key=False): #(x,y,w,h)
    if key:  # 将中心宽高坐标转换为左上右下坐标
        pred[..., 0:2] = pred[..., 0:2] - 1 / 2 * pred[..., 2:4]
        pred[..., 2:4] = pred[..., 0:2] + pred[..., 2:4]
        true[..., 0:2] = true[..., 0:2] - 1 / 2 * true[..., 2:4]
        true[..., 2:4] = true[..., 0:2] + true[..., 2:4]
    x1=torch.max(pred[...,0],true[...,0])
    y1=torch.max(pred[...,1],true[...,1])
    x2=torch.min(pred[...,2],true[...,2])
    y2=torch.min(pred[...,3],true[...,3])
    if len(pred.shape) == 1:
        intersection=max(x2-x1,0)*max(y2-y1,0)
        union=(pred[2]-pred[0])*(pred[3]-pred[1])+(true[2]-true[0])*(true[3]-true[1])-intersection
    else:
        zeros = torch.zeros(len(pred)).to(pred.device.type)
        intersection=torch.max(x2-x1,zeros)*torch.max(y2-y1,zeros)
        union=(pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])+(true[:,2]-true[:,0])*(true[:,3]-true[:,1])-intersection
    return intersection/union




if __name__ == '__main__':
    pred=torch.tensor([[10., 10., 20., 20.],[10., 10., 20., 20.]])
    true=torch.tensor([[10., 10., 20., 20.],[10., 10., 20., 20.]])
    print(iou(pred,true))
