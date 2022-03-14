import torch
from utils import save_checkpoint, load_checkpoint
import torch.optim as optim
import config
from torchvision.utils import save_image
from generator_model import Generator
from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
from tqdm import tqdm




def test_fn(gen_H, gen_Z, loader):

    loop = tqdm(loader, leave=True)

    for idx, (image0, image1) in enumerate(loop):
        image = image0.to(config.DEVICE)
    
        with torch.cuda.amp.autocast():

            fake_image = gen_Z(image)

            save_image(fake_image*0.5+0.5, f"horse2zebrav3/braba.png")




def main():
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )


    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )


    dataset = HorseZebraDataset(root_zebra='./horse2zebrav3/test', root_horse='./horse2zebrav3/test', transform=config.transforms)

    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )


    test_fn(gen_H, gen_Z, loader)



if __name__ == "__main__":
    main()