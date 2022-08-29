import torch

def model_get(args):
    if args.model_continue:
        try:
            return torch.load(r'../result/'+args.name+'.pkl')
        except:
            pass
    dict_choice={'TSF':'TSF(args)._load()',
                 'OD':'OD(args)._load()',
                 }
    return eval(dict_choice[args.type])

class TSF(object):
    def __init__(self,args):
        self.args=args
    def _load(self):
        dict_model={'LSTM':'self._LSTM()',
                    'LSTM_CNN':'self._LSTM_CNN()',
                    'SCINet': 'self._SCINet()'
                   }
        return eval(dict_model[self.args.model])
    def _LSTM(self):
        from model import LSTM
        return LSTM.LSTM(self.args)
    def _LSTM_CNN(self):
        from model import LSTM_CNN
        return LSTM_CNN.LSTM_CNN(self.args)
    def _SCINet(self):
        from model import SCINet
        return SCINet.SCINet(self.args)

class OD(object):
    def __init__(self,args):
        self.args=args
    def _load(self):
        dict_model={'YOLO5':'self._YOLO5()',
                   }
        return eval(dict_model[self.args.model])
    def _YOLO5(self):
        from model import YOLO5
        return YOLO5.YOLO5(self.args)