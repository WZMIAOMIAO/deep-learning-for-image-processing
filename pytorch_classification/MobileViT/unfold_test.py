import time
import torch

batch_size = 8
in_channels = 32
patch_h = 2
patch_w = 2
num_patch_h = 16
num_patch_w = 16
num_patches = num_patch_h * num_patch_w
patch_area = patch_h * patch_w


def official(x: torch.Tensor):
    # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
    x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
    # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
    x = x.transpose(1, 2)
    # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    x = x.reshape(batch_size, in_channels, num_patches, patch_area)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(batch_size * patch_area, num_patches, -1)

    return x


def my_self(x: torch.Tensor):
    # [B, C, H, W] -> [B, C, n_h, p_h, n_w, p_w]
    x = x.reshape(batch_size, in_channels, num_patch_h, patch_h, num_patch_w, patch_w)
    # [B, C, n_h, p_h, n_w, p_w] -> [B, C, n_h, n_w, p_h, p_w]
    x = x.transpose(3, 4)
    # [B, C, n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    x = x.reshape(batch_size, in_channels, num_patches, patch_area)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(batch_size * patch_area, num_patches, -1)

    return x


if __name__ == '__main__':
    t = torch.randn(batch_size, in_channels, num_patch_h * patch_h, num_patch_w * patch_w)
    print(torch.equal(official(t), my_self(t)))

    t1 = time.time()
    for _ in range(1000):
        official(t)
    print(f"official time: {time.time() - t1}")

    t1 = time.time()
    for _ in range(1000):
        my_self(t)
    print(f"self time: {time.time() - t1}")
