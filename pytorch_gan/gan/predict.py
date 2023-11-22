import torch

from model import Generator
from utils import save_gen_imgs


def main():
    weights_path = "weights/generator_weights_199.pth"
    latent_dim = 100
    img_shape = [1, 28, 28]
    generate_img_num = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create generator model and load training weights
    model = Generator(latent_dim=latent_dim, img_shape=img_shape)
    weights_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        # create noise as generator input
        noise = torch.randn(size=(generate_img_num, latent_dim), device=device)

        gen_imgs = model(noise)
        save_gen_imgs(gen_imgs, save_num=generate_img_num, save_path="generated_img.jpg")


if __name__ == '__main__':
    main()
