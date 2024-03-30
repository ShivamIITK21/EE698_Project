from torch.cuda import is_available
import torchvision.transforms as transforms
from checkpoint import load_model
from dataloader import CelebA
from torchvision.utils import save_image
from generator import Generator
import torch

# Black, Blond, Male, Young
LABELS = ""
DESIRED = [-1, -1, -1, 0]
ATTRIBUTES = ["Black_Hair", "Blond_Hair", "Male", "Young"]
TARGET = [0, 0, 0, 1]
NUM_SAMPLES = 10
DEVICE = "cpu"
IMAGE_RES = 128
PREFIX = 2

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    transform = transforms.Compose([
            transforms.Resize((IMAGE_RES, IMAGE_RES)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    gen = Generator(4).to(DEVICE)
    opt = torch.optim.Adam(gen.parameters())
    load_model(gen, opt, "./generator.pth", DEVICE, 0.01)
    data = CelebA("../CelebA", labels=ATTRIBUTES, transform=transform)
    found = 0
    for idx in range(0, len(data)):
        attrs = data.getLabelOnly(idx)        
        valid = True
        for i, val in enumerate(DESIRED):
            if(val == 1):
                if(attrs[i] != 1.0):
                    valid = False
            if(val == 0):
                if(attrs[i] != 0.0):
                    valid = False
            
        if valid :
            found += 1
        
            img = data.getImage(idx)
            img = img.to(DEVICE)
            img = img.view(1, img.size(0), img.size(1), img.size(2))
            target = torch.FloatTensor(TARGET).to(DEVICE)
            target = target.view(1, target.size(0))

            out = gen(img, target)
            save_image(img[0]*0.5 + 0.5, f"""./test_images/{PREFIX}_{idx}_0.png""")
            save_image(out[0]*0.5 + 0.5, f"""./test_images/{PREFIX}_{idx}_1.png""")
            
            
            
            if(found == NUM_SAMPLES):
                exit(0)


