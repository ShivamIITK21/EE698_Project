import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CelebA
from generator import Generator
from discriminator import Discriminator
import torch

DEVICE = "cpu"
ATTRIBUTES = ["Black_Hair", "Blond_Hair", "Male", "Young"]
BATCH_SIZE = 16
IMAGE_RES = 128


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"

    transforms = transforms.Compose([
            transforms.Resize((IMAGE_RES, IMAGE_RES)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    dataset = CelebA(root="../CelebA", labels=ATTRIBUTES, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    gen = Generator(num_attributes=len(ATTRIBUTES))
    disc = Discriminator(image_dim=IMAGE_RES, num_attributes=len(ATTRIBUTES))

    gen = gen.to(DEVICE)
    disc = disc.to(DEVICE)
