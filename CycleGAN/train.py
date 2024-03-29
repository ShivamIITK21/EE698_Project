from PIL.Image import DecompressionBombError
import torchvision.transforms as transforms
from dataloader import HairColorDataset
from torch.utils.data import DataLoader
from generator import Generator
from torchvision.utils import save_image
from discriminator import Discriminator
from tqdm import tqdm
import torch.nn as nn
from checkpoint import checkpoint_model
import torch


BATCH_SIZE = 32 
IMAGE_RES = 128 
LOAD = False
CHECKPOINT = True
CHECKPOINT_PATH = "./checkpoint/checkpoint.pth"
DEVICE = "cpu"
LR = 0.0002
EPOCHS = 100
LAMBDA = 10

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using GPU...")
        DEVICE = "cuda"
    transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    dataset = HairColorDataset(root="./data", transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


    gen_black = Generator()
    gen_blond = Generator()
    disc_black = Discriminator()
    disc_blond = Discriminator()

    gen_black = gen_black.to(DEVICE)
    gen_blond = gen_blond.to(DEVICE)
    disc_black = disc_black.to(DEVICE)
    disc_blond = disc_blond.to(DEVICE)

    opt_gen = torch.optim.Adam(list(gen_black.parameters()) + list(gen_blond.parameters()), lr=LR, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(list(disc_black.parameters()) + list(disc_blond.parameters()), lr=LR, betas=(0.5, 0.999))

    l1 = nn.L1Loss()
    mse= nn.MSELoss()
    
    for epoch in range(0, EPOCHS):
        for idx, (black, blond) in enumerate(tqdm(dataloader)):
            black = black.to(DEVICE)
            blond = blond.to(DEVICE)
            
            # loss for black disc
            fake_black = gen_black(blond)
            black_fake_prob = disc_black(fake_black.detach()) 
            black_real_prob = disc_black(black)
            black_real_loss = mse(black_real_prob, torch.ones_like(black_real_prob))
            black_fake_loss = mse(black_fake_prob, torch.zeros_like(black_fake_prob))
            black_loss = (black_real_loss + black_fake_loss)

            # loss for blond disc
            fake_blond = gen_blond(black)
            blond_fake_prob = disc_blond(fake_blond.detach()) 
            blond_real_prob = disc_blond(blond)
            blond_real_loss = mse(blond_real_prob, torch.ones_like(blond_real_prob))
            blond_fake_loss = mse(blond_fake_prob, torch.zeros_like(blond_fake_prob))
            blond_loss = (blond_real_loss + blond_fake_loss)
            
            disc_loss = (black_loss + blond_loss)/2

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            #loss for gens
            black_fake_prob = disc_black(fake_black)
            blond_fake_prob = disc_blond(fake_blond)
            gen_loss_adv = mse(black_fake_prob, torch.ones_like(black_fake_prob)) + mse(blond_fake_prob, torch.ones_like(blond_fake_prob))
            
            cycle_black = gen_black(fake_blond)
            cycle_blond = gen_blond(fake_black)
            cycle_black_loss = l1(black, cycle_black)
            cycle_blond_loss = l1(blond, cycle_blond)
            gen_cycle_loss = cycle_black_loss + cycle_blond_loss

            identity_black = gen_black(black)
            identity_blond = gen_blond(blond)
            identity_black_loss = l1(black, identity_black)
            identity_blond_loss = l1(blond, identity_blond)
            gen_identity_loss = identity_black_loss + identity_blond_loss
            
            gen_loss = gen_loss_adv + LAMBDA*gen_cycle_loss + 0.2*LAMBDA*gen_identity_loss
            
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            if idx % 100 == 0:
                save_image(black[0] * 0.5 + 0.5, f"saved_images/black_{epoch}_{idx}_1.png")
                save_image(fake_blond[0] * 0.5 + 0.5, f"saved_images/black_{epoch}_{idx}_2.png")
                save_image(cycle_black[0] * 0.5 + 0.5, f"saved_images/black_{epoch}_{idx}_3.png")
                save_image(blond[0] * 0.5 + 0.5, f"saved_images/blond_{epoch}_{idx}_1.png")
                save_image(fake_black[0] * 0.5 + 0.5, f"saved_images/blond_{epoch}_{idx}_2.png")
                save_image(cycle_blond[0] * 0.5 + 0.5, f"saved_images/blond_{epoch}_{idx}_3.png")

        try:
            checkpoint_model(gen_black, opt_gen, epoch, "./gen_black.pth") 
            checkpoint_model(gen_blond, opt_gen, epoch, "./gen_blond.pth") 
            checkpoint_model(disc_black, opt_disc, epoch, "./disc_black.pth") 
            checkpoint_model(disc_blond, opt_disc, epoch, "./disck_blond.pth") 
        except:
            print("Error in saving")
