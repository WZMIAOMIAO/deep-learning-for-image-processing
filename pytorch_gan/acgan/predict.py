import torch

from model import Generator
from utils import save_gen_imgs


def main():
    weights_path = "weights/generator_weights_199.pth"
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
        # create noise and label as generator input
        labels = [0] * 10  # [0, 0, 1, 1, 2, 2, 3, 3]
        num_label = len(labels)
        noise = torch.randn(size=(num_label, latent_dim), device=device)
        gen_labels = torch.as_tensor(labels, dtype=torch.int64, device=device)

        gen_imgs = model(noise, gen_labels)
        save_gen_imgs(gen_imgs, save_num=num_label, save_path="generated_img.jpg")


if __name__ == '__main__':
    main()
