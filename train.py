import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import *
from dataset import BuildDataset, BuildDataloader
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
wandb.init(project="FINESSE - exp1", entity="rashmip")


def main():
    with open("config.yaml", "r") as stream:
        params_loaded = yaml.safe_load(stream)
    
    device = params_loaded['device']
    epochs = params_loaded['train']['epochs']
    exp_dir = params_loaded['save']

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # pixel_criterion = nn.L1Loss().to(device)
    # content_criterion = ContentLoss().to(device)

    pixel_wt = params_loaded['train']['pixel_wt']
    content_wt = params_loaded['train']['content_wt']
    adv_wt = params_loaded['train']['adv_wt']

    d_optimizer = optim.SGD(discriminator.parameters(), 0.0001, 0.9)  # Discriminator learning rate during adversarial network training.
    g_optimizer = optim.Adam(generator.parameters(), 0.0001, (0.9, 0.999))  # Generator learning rate during adversarial network training.

    milestones = [epochs * 0.125, epochs * 0.250, epochs * 0.500, epochs * 0.750]
    d_scheduler = MultiStepLR(d_optimizer, list(map(int, milestones)), 0.5)         # Discriminator model scheduler during adversarial training.
    g_scheduler = MultiStepLR(g_optimizer, list(map(int, milestones)), 0.5)         # Generator model scheduler during adversarial training.

    dataset = BuildDataset(params_loaded)
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    val_size = full_size - train_size 
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_build_loader = BuildDataloader(trainset, batch_size=params_loaded['train']['batch_size'], shuffle=True, num_workers=2)
    train_dataloader = train_build_loader.loader()
    val_build_loader = BuildDataloader(valset, batch_size=params_loaded['train']['batch_size'], shuffle=True, num_workers=2)
    val_dataloader = val_build_loader.loader()
    
    if params_loaded['resume']:
        print("Resuming...")
        discriminator.load_state_dict(torch.load(params_loaded['resume_d_weight']))
        generator.load_state_dict(torch.load(params_loaded['resume_g_weight']))
    
    for epoch in tqdm(range(params_loaded['train']['start_epoch'], epochs)):
        batches = len(train_dataloader)
        discriminator.train()
        generator.train()
        for index, batch in enumerate(train_dataloader):
            img = batch['images'].to(device)
            embedding = batch['embedding'].to(device)
            noise = torch.normal(0,1,size=(params_loaded['train']['batch_size'], params_loaded['train']['noise_size']),dtype=torch.float32)
            noise = noise.to(device)
            fake_embed = embedding.detach().clone()
            indexes = torch.randperm(fake_embed.shape[0])
            fake_embed = fake_embed[indexes].to(device)
            
            discriminator.zero_grad()
            
            gen = generator(noise, embedding)
            real_image_real_text = discriminator(img,embedding)
            real_image_fake_text = discriminator(img,fake_embed)
            fake_image_real_text = discriminator(gen,embedding)
            
            d_loss = discriminator.discriminator_loss(real_image_real_text, fake_image_real_text, real_image_fake_text)
            d_loss.backward()
            d_optimizer.step()

            generator.zero_grad()
            gen = generator(noise, embedding)
            real_image_real_text = discriminator(img,embedding)
            real_image_fake_text = discriminator(img,fake_embed)
            fake_image_real_text = discriminator(gen,embedding)
            
            # pixel_loss = pixel_criterion(gen, img.detach())
            # content_loss = content_criterion(gen, img.detach())
            adv_loss = generator.generator_loss(fake_image_real_text)

            g_loss =  adv_wt*adv_loss #+pixel_wt*pixel_loss +content_wt*content_loss
            g_loss.backward()
            g_optimizer.step()

            if (index + 1) % 10 == 0 or (index + 1) == batches:
                print(f"Train stage: adversarial "
                    f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                    f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
                    f"Adv Loss: {adv_loss:.4f}")# Pixel Loss: {pixel_loss} Content loss: {content_loss}.")
        
        wandb.log({"D Loss": d_loss, "G Loss": g_loss, "Adversarial Loss": adv_loss})#, "Pixel Loss":pixel_loss})#, "Content Loss":content_loss})
        torch.save(discriminator.state_dict(), os.path.join(exp_dir, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir, f"g_epoch{epoch + 1}.pth"))
        
        # Adjust the learning rate of the adversarial model.
        d_scheduler.step()
        g_scheduler.step()

        # Validation Step
        batches = len(val_dataloader)
        # Set generator model in verification mode.
        generator.eval()
        with torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                img = batch['images'].to(device)
                embedding = batch['embedding'].to(device)
                noise = torch.normal((params_loaded['train']['batch_size'], params_loaded['train']['noise_size']),dtype=torch.float32)
                noise = noise.to(device)
                gen = generator(noise,embedding)

                gen = 0.5 * gen.detach() + 0.5
                image_array = np.array(gen)
                output_path = params_loaded['results']
  

                filename = os.path.join(output_path,batch['names'])
                im = Image.fromarray(image_array)
                im.save(filename)
                
                
    # Save the weight of the adversarial model under the last Epoch in this stage.
    torch.save(discriminator.state_dict(), os.path.join(exp_dir, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir, "g-last.pth"))

if __name__ == "__main__":
    main()
