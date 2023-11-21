import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Generator, Discriminator
from utils import save_gen_imgs, make_dirs


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch GAN Training", add_help=add_help)
    parser.add_argument("--device", type=str, default="cuda:0", help="training device, e.g. cpu, cuda:0")
    parser.add_argument("--save_weights_dir", type=str, default="./weights", help="save dir for model weights")
    parser.add_argument("--save_imgs_dir", type=str, default="./gen_imgs", help="save dir for generated imgs")
    parser.add_argument("--save_freq", type=int, default=20, help="save frequency for weights and generated imgs")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers, default: 8")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_shape", type=int, nargs=3, default=[1, 28, 28], help="image shape: C, H, W")

    return parser


def main(args):
    torch.manual_seed(1234)
    save_weights_dir = args.save_weights_dir
    save_imgs_dir = args.save_imgs_dir
    save_freq = args.save_freq
    make_dirs(save_weights_dir)
    make_dirs(save_imgs_dir)

    # create generator and discriminator model
    device = torch.device(args.device)
    img_shape = args.img_shape  # [C, H, W]
    generator = Generator(latent_dim=args.latent_dim, img_shape=img_shape)
    generator.to(device)
    discriminator = Discriminator(img_shape=img_shape)
    discriminator.to(device)

    # config dataset and dataloader
    transform = transforms.Compose([transforms.Resize(img_shape[1:]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    mnist_dataset = datasets.MNIST(root="./mnist_folder",
                                   train=True,
                                   download=True,
                                   transform=transform)
    
    dataloader = DataLoader(dataset=mnist_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True)
    
    # define loss function
    adversarial_loss = nn.BCELoss()

    # define optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.epochs):
        g_loss_accumulator = 0.
        d_loss_accumulator = 0.
        for step, (imgs, _) in enumerate(tqdm(dataloader, file=sys.stdout)):
            real_imgs = imgs.to(device)
            b = imgs.shape[0]
            # adversarial ground truths
            valid = torch.ones(size=(b, 1), device=device)
            fake = torch.zeros(size=(b, 1), device=device)
            
            # create noise as generator input
            noise = torch.randn(size=(b, args.latent_dim), device=device)

            # train generator
            optimizer_g.zero_grad()
            gen_imgs = generator(noise)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()
            g_loss_accumulator += g_loss.item()

            # train discriminator
            optimizer_d.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            d_loss_accumulator += d_loss.item()
        
        g_loss_mean = g_loss_accumulator / (step + 1)
        d_loss_mean = d_loss_accumulator / (step + 1)
        print(f"[{epoch + 1}/{args.epochs}] g_loss: {g_loss_mean:.3f}, d_loss: {d_loss_mean:.3f}")

        if epoch % save_freq == 0:
            save_gen_imgs(gen_imgs, save_path=os.path.join(save_imgs_dir, f"gen_img_{epoch}.jpg"))
            torch.save(generator.state_dict(), os.path.join(save_weights_dir, f"generator_weights_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_weights_dir, f"discriminator_weights_{epoch}.pth"))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
