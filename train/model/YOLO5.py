#版本:2022.07
#YOLO5: model_param---[第一层输出维度,C3层残差数]---YOLO5n[16,1],YOLO5s[32,1],YOLO5m[48,2],YOLO5l[64,3],YOLO5x[80,4]
#YOLO5: loss_param--- [每层输出权重,3*[边框权重,置信度权重,类别权重]]---[[1/3,1/3,1/3],[0.2,0.5,0.3],[0.3,0.4,0.3],[0.4,0.4,0.3]]
import torch

class conv2d_bn_silu(torch.nn.Module):
    def __init__(self,in1,out1,kernel_size,stride):
        super().__init__()
        self.conv2d=torch.nn.Conv2d(in1,out1,kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.bn=torch.nn.BatchNorm2d(out1)
        self.silu=torch.nn.SiLU()
    def forward(self,input1):
        x=self.conv2d(input1)
        x=self.bn(x)
        x=self.silu(x)
        return x

class residual(torch.nn.Module):
    def __init__(self,in1):
        super().__init__()
        self.conv2d_bn_silu_1=conv2d_bn_silu(in1,in1,kernel_size=1,stride=1)
        self.conv2d_bn_silu_2 = conv2d_bn_silu(in1,in1, kernel_size=3, stride=1)
    def forward(self,input1):
        x=self.conv2d_bn_silu_1(input1)
        x=self.conv2d_bn_silu_2(x)
        return x+input1

class c3(torch.nn.Module):
    def __init__(self,in1,out1,n):
        super().__init__()
        self.conv2d_bn_silu_1=conv2d_bn_silu(in1,in1//2,kernel_size=1,stride=1)
        self.Sequential_residual=torch.nn.Sequential(*(residual(in1//2) for i in range(n)))
        self.conv2d_bn_silu_2 = conv2d_bn_silu(in1, in1//2, kernel_size=1, stride=1)
        self.conv2d_bn_silu_3 = conv2d_bn_silu(in1, out1, kernel_size=1, stride=1)
    def forward(self,input1):
        input2=self.conv2d_bn_silu_1(input1)
        x=self.Sequential_residual(input2)
        x=x+input2
        input1=self.conv2d_bn_silu_2(input1)
        x=torch.cat([x,input1],axis=1)
        x=self.conv2d_bn_silu_3(x)
        return x

class sppf(torch.nn.Module):
    def __init__(self,in1):
        super().__init__()
        self.conv2d_bn_silu_1 = conv2d_bn_silu(in1, in1//2, kernel_size=1, stride=1)
        self.MaxPool2d_1=torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2,dilation=1)
        self.MaxPool2d_2 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
        self.MaxPool2d_3 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv2d_bn_silu_2 = conv2d_bn_silu(2*in1, in1, kernel_size=1, stride=1)
    def forward(self, input1):
        input1 = self.conv2d_bn_silu_1(input1)
        m1=self.MaxPool2d_1(input1)
        m2=self.MaxPool2d_2(m1)
        m3=self.MaxPool2d_3(m2)
        x=torch.cat([input1,m1,m2,m3],axis=1)
        x=self.conv2d_bn_silu_2(x)
        return x

class YOLO5(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        dimension=args.model_param[0]
        n=args.model_param[1]
        class_=args.OD_class
        self.YOLO5_1 = conv2d_bn_silu(3,dimension,6,2)
        self.YOLO5_2 = conv2d_bn_silu(dimension,2*dimension,3,2)
        self.YOLO5_3 = c3(2*dimension,2*dimension,n)
        self.YOLO5_4 = conv2d_bn_silu(2*dimension,4*dimension,3,2)
        self.YOLO5_5 = c3(4*dimension,4*dimension,2*n)
        self.YOLO5_6 = conv2d_bn_silu(4*dimension,8*dimension,3,2)
        self.YOLO5_7 = c3(8*dimension,8*dimension,3*n)
        self.YOLO5_8 = conv2d_bn_silu(8*dimension,16*dimension,3,2)
        self.YOLO5_9 = c3(16*dimension,16*dimension,n)
        self.YOLO5_10 = sppf(16*dimension)
        self.YOLO5_11 = conv2d_bn_silu(16 * dimension, 8 * dimension, 1, 1)
        self.YOLO5_12 = torch.nn.Upsample(scale_factor=2)
        #concat([YOLO5_7,YOLO5_12],axis=1)
        self.YOLO5_13 = c3(16*dimension,8*dimension,n)
        self.YOLO5_14 = conv2d_bn_silu(8 * dimension, 4 * dimension, 1, 1)
        self.YOLO5_15 = torch.nn.Upsample(scale_factor=2)
        #concat([YOLO5_5,YOLO5_15],axis=1)
        self.YOLO5_16 = c3(8* dimension,4*dimension, n) # output_1
        self.YOLO5_17 = conv2d_bn_silu(4 * dimension, 4 * dimension, 3, 2)
        #concat([YOLO5_14,YOLO5_17],axis=1)
        self.YOLO5_18 = c3(8 * dimension, 8 * dimension, n) # output_2
        self.YOLO5_19 = conv2d_bn_silu(8 * dimension, 8 * dimension, 3, 2)
        #concat([YOLO5_11,YOLO5_19],axis=1)
        self.YOLO5_20 = c3(16 * dimension, 16 * dimension, n) # output_3
        self.output_1 = torch.nn.Conv2d(4 * dimension, 3 * (class_+ 5), kernel_size=1, stride=1,padding=0)
        self.output_sigmoid_1 = torch.nn.Sigmoid()
        self.output_2 = torch.nn.Conv2d(8 * dimension, 3 * (class_+ 5), kernel_size=1, stride=1,padding=0)
        self.output_sigmoid_2 = torch.nn.Sigmoid()
        self.output_3 = torch.nn.Conv2d(16 * dimension, 3 * (class_+ 5), kernel_size=1, stride=1,padding=0)
        self.output_sigmoid_3 = torch.nn.Sigmoid()
    def forward(self,input1):
        img_len = self.args.OD_size
        input1=input1.permute(0,3,1,2)
        x=self.YOLO5_1(input1)
        x = self.YOLO5_2(x)
        x = self.YOLO5_3(x)
        x = self.YOLO5_4(x)
        cat2 = self.YOLO5_5(x)
        x = self.YOLO5_6(cat2)
        cat1 = self.YOLO5_7(x)
        x = self.YOLO5_8(cat1)
        x = self.YOLO5_9(x)
        x = self.YOLO5_10(x)
        cat4 = self.YOLO5_11(x)
        x = self.YOLO5_12(cat4)
        x=torch.cat([x,cat1], axis=1)
        x = self.YOLO5_13(x)
        cat3 = self.YOLO5_14(x)
        x = self.YOLO5_15(cat3)
        x = torch.cat([x, cat2], axis=1)
        x = self.YOLO5_16(x)
        output_1 = self.output_1(x)
        output_1 = self.output_sigmoid_1(output_1)
        x = self.YOLO5_17(x)
        x = torch.cat([x, cat3], axis=1)
        x = self.YOLO5_18(x)
        output_2 = self.output_2(x)
        output_2 = self.output_sigmoid_2(output_2)
        x = self.YOLO5_19(x)
        x = torch.cat([x, cat4], axis=1)
        x = self.YOLO5_20(x)
        output_3 = self.output_3(x)
        output_3 = self.output_sigmoid_3(output_3)
        return [output_1.reshape(-1,3,(self.args.OD_class+ 5),img_len//8,img_len//8).permute(0,1,3,4,2),\
               output_2.reshape(-1,3,(self.args.OD_class+ 5),img_len//16,img_len//16).permute(0,1,3,4,2),\
               output_3.reshape(-1,3,(self.args.OD_class+ 5),img_len//32,img_len//32).permute(0,1,3,4,2)]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='!!!')
    parser.add_argument('--model_param', default=[16, 1], type=list)
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--OD_class', default=1, type=int)
    parser.add_argument('--OD_size', default=640, type=int)
    args = parser.parse_args()
    model=YOLO5(args)
    print(model)
    img1=torch.rand(2,640,640,3)
    output_1,output_2,output_3=model(img1)
    print('1')
