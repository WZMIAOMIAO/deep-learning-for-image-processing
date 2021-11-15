import os
import numpy as np


def parse_model_cfg(path: str):
    # 检查文件是否存在
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise FileNotFoundError("the cfg file not exist...")

    # 读取文件信息
    with open(path, "r") as f:
        lines = f.read().split("\n")

    # 去除空行和注释行
    lines = [x for x in lines if x and not x.startswith("#")]
    # 去除每行开头和结尾的空格符
    lines = [x.strip() for x in lines]

    mdefs = []  # module definitions
    for line in lines:
        if line.startswith("["):  # this marks the start of a new block
            mdefs.append({})
            mdefs[-1]["type"] = line[1:-1].strip()  # 记录module类型
            # 如果是卷积模块，设置默认不使用BN(普通卷积层后面会重写成1，最后的预测层conv保持为0)
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = 0
        else:
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()

            if key == "anchors":
                # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                val = val.replace(" ", "")  # 将空格去除
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                # TODO: .isnumeric() actually fails to get the float case
                if val.isnumeric():  # return int or float 如果是数值的情况
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string  是字符的情况

    # check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    # 遍历检查每个模型的配置
    for x in mdefs[1:]:  # 0对应net配置
        # 遍历每个配置字典中的key值
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields:{} in cfg".format(k))

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
