import re
import torch
from src import u2net_full, u2net_lite

layers = {"encode": [7, 6, 5, 4, 4, 4],
          "decode": [4, 4, 5, 6, 7]}


def convert_conv_bn(new_weight, prefix, ks, v):
    if "conv" in ks[0]:
        if "weight" == ks[1]:
            new_weight[prefix + ".conv.weight"] = v
        elif "bias" == ks[1]:
            new_weight[prefix + ".conv.bias"] = v
        else:
            print(f"unrecognized weight {prefix + ks[1]}")
        return

    if "bn" in ks[0]:
        if "running_mean" == ks[1]:
            new_weight[prefix + ".bn.running_mean"] = v
        elif "running_var" == ks[1]:
            new_weight[prefix + ".bn.running_var"] = v
        elif "weight" == ks[1]:
            new_weight[prefix + ".bn.weight"] = v
        elif "bias" == ks[1]:
            new_weight[prefix + ".bn.bias"] = v
        elif "num_batches_tracked" == ks[1]:
            return
        else:
            print(f"unrecognized weight {prefix + ks[1]}")
        return


def convert(old_weight: dict):
    new_weight = {}
    for k, v in old_weight.items():
        ks = k.split(".")
        if ("stage" in ks[0]) and ("d" not in ks[0]):
            # encode stage
            num = int(re.findall(r'\d', ks[0])[0]) - 1
            prefix = f"encode_modules.{num}"
            if "rebnconvin" == ks[1]:
                # ConvBNReLU module
                prefix += ".conv_in"
                convert_conv_bn(new_weight, prefix, ks[2:], v)
            elif ("rebnconv" in ks[1]) and ("d" not in ks[1]):
                num_ = int(re.findall(r'\d', ks[1])[0]) - 1
                prefix += f".encode_modules.{num_}"
                convert_conv_bn(new_weight, prefix, ks[2:], v)
            elif ("rebnconv" in ks[1]) and ("d" in ks[1]):
                num_ = layers["encode"][num] - int(re.findall(r'\d', ks[1])[0]) - 1
                prefix += f".decode_modules.{num_}"
                convert_conv_bn(new_weight, prefix, ks[2:], v)
            else:
                print(f"unrecognized key: {k}")

        elif ("stage" in ks[0]) and ("d" in ks[0]):
            # decode stage
            num = 5 - int(re.findall(r'\d', ks[0])[0])
            prefix = f"decode_modules.{num}"
            if "rebnconvin" == ks[1]:
                # ConvBNReLU module
                prefix += ".conv_in"
                convert_conv_bn(new_weight, prefix, ks[2:], v)
            elif ("rebnconv" in ks[1]) and ("d" not in ks[1]):
                num_ = int(re.findall(r'\d', ks[1])[0]) - 1
                prefix += f".encode_modules.{num_}"
                convert_conv_bn(new_weight, prefix, ks[2:], v)
            elif ("rebnconv" in ks[1]) and ("d" in ks[1]):
                num_ = layers["decode"][num] - int(re.findall(r'\d', ks[1])[0]) - 1
                prefix += f".decode_modules.{num_}"
                convert_conv_bn(new_weight, prefix, ks[2:], v)
            else:
                print(f"unrecognized key: {k}")
        elif "side" in ks[0]:
            # side
            num = 6 - int(re.findall(r'\d', ks[0])[0])
            prefix = f"side_modules.{num}"
            if "weight" == ks[1]:
                new_weight[prefix + ".weight"] = v
            elif "bias" == ks[1]:
                new_weight[prefix + ".bias"] = v
            else:
                print(f"unrecognized weight {prefix + ks[1]}")
        elif "outconv" in ks[0]:
            prefix = f"out_conv"
            if "weight" == ks[1]:
                new_weight[prefix + ".weight"] = v
            elif "bias" == ks[1]:
                new_weight[prefix + ".bias"] = v
            else:
                print(f"unrecognized weight {prefix + ks[1]}")
        else:
            print(f"unrecognized key: {k}")

    return new_weight


def main_1():
    from u2net import U2NET, U2NETP

    old_m = U2NET()
    old_m.load_state_dict(torch.load("u2net.pth", map_location='cpu'))
    new_m = u2net_full()

    # old_m = U2NETP()
    # old_m.load_state_dict(torch.load("u2netp.pth", map_location='cpu'))
    # new_m = u2net_lite()

    old_w = old_m.state_dict()

    w = convert(old_w)
    new_m.load_state_dict(w, strict=True)

    torch.random.manual_seed(0)
    x = torch.randn(1, 3, 288, 288)
    old_m.eval()
    new_m.eval()
    with torch.no_grad():
        out1 = old_m(x)[0]
        out2 = new_m(x)
        assert torch.equal(out1, out2)
        torch.save(new_m.state_dict(), "u2net_full.pth")


def main():
    old_w = torch.load("u2net.pth", map_location='cpu')
    new_m = u2net_full()

    # old_w = torch.load("u2netp.pth", map_location='cpu')
    # new_m = u2net_lite()

    w = convert(old_w)
    new_m.load_state_dict(w, strict=True)
    torch.save(new_m.state_dict(), "u2net_full.pth")


if __name__ == '__main__':
    main()
