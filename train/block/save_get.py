import torch
def save_get(args,model):
    if args.model_save[0]:
        dict_choice={'pkl':'save(args)._pkl(model)'
                    }
        return eval(dict_choice[args.model_save[1]])

class save(object):
    def __init__(self,args):
        self.args=args
    def _pkl(self,model):
        torch.save(model,'../result/'+self.args.name+'.pkl')
        print('| 模型已保存 |')
