import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CelebA
from generator import Generator
import torch.nn as nn
from discriminator import Discriminator
from torchvision.utils import save_image
from tqdm import tqdm
import torch

DEVICE = "cpu"
ATTRIBUTES = ["Black_Hair", "Blond_Hair", "Male", "Young"]
BATCH_SIZE = 16
IMAGE_RES = 128
LR = 1e-4
EPOCHS = 100
CLS_LAMBDA = 1
REC_LAMBDA = 10
GRAD_PENALTY = 10
N_CRITIC = 5

def gradient_penalty(y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(DEVICE)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


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

    opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for epoch in range(0, EPOCHS):
        for idx, (real, attr_real) in enumerate(tqdm(dataloader)):
            real = real.to(DEVICE)
            attr_real = attr_real.to(DEVICE)

            # generate target attr
            rand_idx = torch.randperm(real.size(0))
            attr_target = attr_real[rand_idx].clone().to(DEVICE)

            # Computing Disc loss
            real_img_logits, real_attr_logits = disc(real)
            disc_loss_real = -1*real_img_logits.mean()


            fake_image = gen(real, attr_target)
            fake_image_logits, fake_attr_logits = disc(fake_image.detach())
            disc_loss_fake = fake_image_logits.mean()

            disc_classification_loss = bce(real_attr_logits, attr_real)

            alpha = torch.rand(real.size(0), 1, 1, 1).to(DEVICE)
            real_hat = (alpha*real + (1-alpha)*fake_image).requires_grad_(True)
            disc_hat_logits, _ = disc(real_hat)
            disc_loss_gp = gradient_penalty(disc_hat_logits, real_hat)


            disc_loss = disc_loss_real + disc_loss_fake + GRAD_PENALTY*disc_loss_gp + CLS_LAMBDA*disc_classification_loss


            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()


            # Computing Gen Loss
            if (idx+1)%N_CRITIC == 0:
                fake_img = gen(real, attr_target)
                fake_image_logits, fake_attr_logits = disc(fake_img)
                gen_fake_loss = -1*fake_image_logits.mean()


                reconstruction = gen(fake_img, attr_real)
                gen_reconstruction_loss = l1(reconstruction, real)

                gen_classification_loss = bce(fake_attr_logits, attr_target)

                gen_loss = gen_fake_loss + CLS_LAMBDA*gen_classification_loss + REC_LAMBDA*gen_reconstruction_loss
                opt_gen.zero_grad()
                gen_loss.backward()
                opt_gen.step()

            if idx%200 == 0:
                with torch.no_grad():
                    save_image(real[0]*0.5 + 0.5, f"""./saved_images/{epoch}_{idx}.png""")
                    image = real[0].clone().to(DEVICE)
                    image = image.view(1, image.size(0), image.size(1), image.size(2))
                    for i, attr in enumerate(ATTRIBUTES):
                        target = torch.FloatTensor([float(x==i) for x in range(0, len(ATTRIBUTES))]).to(DEVICE)
                        target = target.view(1, target.size(0))
                        out = gen(image, target)
                        save_image(out[0]*0.5 + 0.5, f"""./saved_images/{epoch}_{idx}_{i}{attr}.png""")
                    

