import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Generator, Discriminator
from utils import save_gen_imgs, make_dirs, create_logger, record_args


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch GAN Training", add_help=add_help)
    parser.add_argument("--device", type=str, default="cuda:0", help="training device, e.g. cpu, cuda:0")
    parser.add_argument("--save_weights_dir", type=str, default="./weights", help="save dir for model weights")
    parser.add_argument("--save_imgs_dir", type=str, default="./gen_imgs", help="save dir for generated imgs")
    parser.add_argument("--save_freq", type=int, default=20, help="save frequency for weights and generated imgs")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes, default: 10")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers, default: 8")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_shape", type=int, nargs=3, default=[3, 32, 32], help="image shape: C, H, W")

    return parser


def main(args):
    torch.manual_seed(1234)
    logger = create_logger()
    record_args(logger, args)

    save_weights_dir = args.save_weights_dir
    save_imgs_dir = args.save_imgs_dir
    save_freq = args.save_freq
    make_dirs(save_weights_dir)
    make_dirs(save_imgs_dir)

    if "cuda" in args.device and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"using device: {device} for training.")

    # create generator and discriminator model
    num_classes = args.num_classes
    img_shape = args.img_shape  # [C, H, W]
    generator = Generator(num_classes=num_classes, latent_dim=args.latent_dim, img_shape=img_shape)
    generator.to(device)
    logger.info(str(generator))

    discriminator = Discriminator(num_classes=num_classes, img_shape=img_shape)
    discriminator.to(device)
    logger.info(str(discriminator))

    # config dataset and dataloader
    transform = transforms.Compose([transforms.Resize(img_shape[1:]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    cifar10_dataset = datasets.CIFAR10(root="./cifar10_folder",
                                       train=True,
                                       download=True,
                                       transform=transform)
    logger.info(f"cifar10 classes: {cifar10_dataset.classes}")

    dataloader = DataLoader(dataset=cifar10_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True)

    # define loss function
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    # define optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        g_loss_accumulator = 0.
        d_loss_accumulator = 0.
        for step, (imgs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)
            b = imgs.shape[0]
            # adversarial ground truths
            valid = torch.ones(size=(b, 1), device=device)
            fake = torch.zeros(size=(b, 1), device=device)

            # create noise and label as generator input
            noise = torch.randn(size=(b, args.latent_dim), device=device)
            gen_labels = torch.randint(0, num_classes, size=(b,), dtype=torch.int64, device=device)

            # train generator
            optimizer_g.zero_grad()
            gen_imgs = generator(noise, gen_labels)
            validity, pred_labels = discriminator(gen_imgs)
            adver_loss = adversarial_loss(validity, valid)
            aux_loss = auxiliary_loss(pred_labels, gen_labels)
            g_loss = (adver_loss + aux_loss) / 2
            g_loss.backward()
            optimizer_g.step()
            g_loss_accumulator += g_loss.item()

            # train discriminator
            optimizer_d.zero_grad()
            # loss for real images
            validity, pred_labels = discriminator(real_imgs)
            real_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_labels, real_labels))
            # loss for fake images
            validity, pred_labels = discriminator(gen_imgs.detach())
            fake_loss = (adversarial_loss(validity, fake) + auxiliary_loss(pred_labels, gen_labels))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            d_loss_accumulator += d_loss.item()

        g_loss_mean = g_loss_accumulator / (step + 1)
        d_loss_mean = d_loss_accumulator / (step + 1)
        logger.info(f"[{epoch + 1}/{args.epochs}] g_loss: {g_loss_mean:.3f}, d_loss: {d_loss_mean:.3f}")

        if epoch % save_freq == 0 or epoch == args.epochs - 1:
            torch.save(generator.state_dict(), os.path.join(save_weights_dir, f"generator_weights_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_weights_dir, f"discriminator_weights_{epoch}.pth"))

            generator.eval()
            with torch.inference_mode():
                # create noise and label as generator input
                noise = torch.randn(size=(num_classes, args.latent_dim), device=device)
                gen_labels = torch.arange(0, num_classes, dtype=torch.int64, device=device)
                gen_imgs = generator(noise, gen_labels)
                save_gen_imgs(gen_imgs, save_num=10, save_path=os.path.join(save_imgs_dir, f"gen_img_{epoch}.jpg"))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
