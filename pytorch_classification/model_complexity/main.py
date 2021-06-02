import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from prettytable import PrettyTable
from model import efficientnetv2_s


def main():
    model = efficientnetv2_s()

    # option1
    for name, para in model.named_parameters():
        # 除head外，其他权重全部冻结
        if "head" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))

    complexity = model.complexity(224, 224, 3)
    table = PrettyTable()
    table.field_names = ["params", "freeze-params", "train-params", "FLOPs", "acts"]
    table.add_row([complexity["params"],
                   complexity["freeze"],
                   complexity["params"] - complexity["freeze"],
                   complexity["flops"],
                   complexity["acts"]])
    print(table)

    # option2
    tensor = (torch.rand(1, 3, 224, 224),)
    flops = FlopCountAnalysis(model, tensor)
    print(flops.total())

    print(parameter_count_table(model))


if __name__ == '__main__':
    main()
