import pandas as pd
import shutil

seps = ["Black_Hair", "Blond_Hair"]
fnames = ["black", "blond"]

attr = pd.read_csv("./CelebA/list_attr_celeba.csv")
for (c, f) in zip(seps, fnames):
    imgs = attr.loc[attr[c] == 1, 'image_id'].tolist()
    for img in imgs:
        print("Copying...")
        shutil.copy(f"""./CelebA/Img/img_align_celeba/{img}""", f"""./CycleGAN/data/{f}/{img}""")
         
print(attr.head(5))

