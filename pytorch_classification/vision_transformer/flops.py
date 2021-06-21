import torch
from fvcore.nn import FlopCountAnalysis

from vit_model import Attention


def main():
    # Self-Attention
    a1 = Attention(dim=512, num_heads=1)
    a1.proj = torch.nn.Identity()  # remove Wo

    # Multi-Head Attention
    a2 = Attention(dim=512, num_heads=8)

    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512),)

    flops1 = FlopCountAnalysis(a1, t)
    print("Self-Attention FLOPs:", flops1.total())

    flops2 = FlopCountAnalysis(a2, t)
    print("Multi-Head Attention FLOPs:", flops2.total())


if __name__ == '__main__':
    main()

