import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from prettytable import PrettyTable
from torchvision.models import resnet50
from model import efficientnetv2_s
import numpy as np

def flops_fvcore():
    model = efficientnetv2_s(num_classes=6)

    # option1
    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))
    #
    # complexity = model.complexity(224, 224, 3)
    # table = PrettyTable()
    # table.field_names = ["params", "freeze-params", "train-params", "FLOPs", "acts"]
    # table.add_row([complexity["params"],
    #                complexity["freeze"],
    #                complexity["params"] - complexity["freeze"],
    #                complexity["flops"],
    #                complexity["acts"]])
    # print(table)

    # option2
    tensor = (torch.rand(1, 3, 224, 224),)
    # 分析FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total())
    # 分析parameters
    print(parameter_count_table(model))

def flops_torchstat():
    '''
    在PyTorch中，可以使用torchstat这个库来查看网络模型的一些信息，包括总的参数量params、MAdd、显卡内存占用量和FLOPs等
    pip install torchstat
    '''
    from torchstat import stat
    from torchvision.models import resnet50
    model = efficientnetv2_s(num_classes=6)
    stat(model, (3, 224, 224))

def flops_thop():
    '''
    code by zzg-2020-05-19
    pip install thop
    '''
    import torch
    from thop import profile

    device = torch.device("cpu")
    # input_shape of model,batch_size=1
    net = efficientnetv2_s(num_classes=6)  ##定义好的网络模型

    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(input,))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

def flops_ptflops():
    # pip install ptflops
    from ptflops import get_model_complexity_info
    from torchvision.models import resnet50
    model = efficientnetv2_s(num_classes=6)
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)

if __name__ == '__main__':
    flops_ptflops()
