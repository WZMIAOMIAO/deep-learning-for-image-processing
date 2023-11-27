import torch
from torchvision.utils import save_image

from model import Generator


def main():
    weights_path = "weights/generator_weights_60.pth"
    latent_dim = 100
    img_shape = [3, 32, 32]
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create generator model and load training weights
    model = Generator(num_classes=num_classes, latent_dim=latent_dim, img_shape=img_shape)
    weights_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        imgs_list = []
        for _ in range(10):
            # create noise and label as generator input
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            num_label = len(labels)
            noise = torch.randn(size=(num_label, latent_dim), device=device)
            gen_labels = torch.as_tensor(labels, dtype=torch.int64, device=device)

            # [10, 3, 32, 32]
            gen_imgs = model(noise, gen_labels)
            # [10, 3, 32, 32] -> [1, 3, 32, 320]
            imgs = torch.concat(gen_imgs.chunk(chunks=num_label, dim=0), dim=3)
            imgs_list.append(imgs)

        # [1, 3, 320, 320]
        imgs = torch.concat(imgs_list, dim=2)
        imgs.mul_(0.5).add_(0.5)
        save_image(imgs, fp="generated_img.jpg")


if __name__ == '__main__':
    main()
