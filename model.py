import torch.nn as nn
import torch
import torchvision.models as models



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape = nn.Sequential(nn.Linear(228,4*4*256), nn.LeakyReLU())
        self.embedding_encoder = nn.Sequential(nn.Linear(300,128),nn.LeakyReLU())
        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.LazyConvTranspose2d(256,kernel_size=5,padding=2), nn.InstanceNorm2d(256), nn.LeakyReLU())
        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.LazyConvTranspose2d(128,kernel_size=5,padding=2), nn.InstanceNorm2d(128), nn.LeakyReLU())
        self.upsample3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.LazyConvTranspose2d(128,kernel_size=5,padding=2), nn.InstanceNorm2d(128), nn.LeakyReLU())
        self.output_conv = nn.Sequential(nn.LazyConvTranspose2d(3, kernel_size=3,padding=1), nn.Tanh())
        self.cross_entropy = nn.CrossEntropyLoss()
        self._init_weights()
    
    def init_weights(self, l):
      if type(l)==nn.Linear:
        nn.init.normal_(l.weight, mean=0.0, std=0.01)
        if l.bias != None:
            l.bias.data.fill_(0)

    def _init_weights(self):
        self.embedding_encoder.apply(self.init_weights)
        self.upsample1.apply(self.init_weights)
        self.upsample2.apply(self.init_weights)
        self.output_conv.apply(self.init_weights)

    def generator_loss(self,fake_output):
        return self.cross_entropy(torch.ones_like(fake_output), fake_output)

    def forward(self,noise, embedding):
       out = self.embedding_encoder(embedding)
       out = torch.cat((noise,out), axis=1) #axis=1
       out = self.reshape(out)
       out = torch.reshape(out, (-1,256,4,4))
       out = self.upsample1(out)
       out = self.upsample1(out)
       out = self.upsample2(out)
       out = self.upsample3(out)
       out = self.output_conv(out)
       return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.LazyConv2d(32,kernel_size=4,stride=2,padding=1), nn.LeakyReLU(),nn.Dropout(p=0.25))
        self.conv2 = nn.Sequential(nn.LazyConv2d(64,kernel_size=4,stride=2,padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(),nn.Dropout(p=0.25))
        self.conv3 = nn.Sequential(nn.LazyConv2d(128,kernel_size=4,stride=2,padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(),nn.Dropout(p=0.25))
        self.conv4 = nn.Sequential(nn.LazyConv2d(256,kernel_size=4,stride=2,padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU())
        self.embedding_encoder = nn.Sequential(nn.Linear(300,128),nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.LazyConv2d(512,kernel_size=4,stride=2,padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(),nn.Dropout(p=0.25))
        self.output_layer = nn.Sequential(nn.Flatten(), nn.Linear(512*2*2,1))

        self.cross_entropy = nn.BCEWithLogitsLoss()
    
    def discriminator_loss(self,real_image_real_text, fake_image_real_text, real_image_fake_text):
        real_loss = self.cross_entropy(torch.ones_like(real_image_real_text), real_image_real_text)
        fake_loss = (self.cross_entropy(torch.zeros_like(fake_image_real_text), fake_image_real_text) + 
                    self.cross_entropy(torch.zeros_like(real_image_fake_text), real_image_fake_text))/2

        total_loss = real_loss + fake_loss
        return total_loss

    def apply_spectral_norm(self):
      for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d, nn.InstanceNorm1d)):
            nn.utils.spectral_norm(m)

    def forward(self,img, embedding):
       out = self.conv1(img)
       out = self.conv2(out)
       out = self.conv3(out)
       out = self.conv4(out)
       embed = self.embedding_encoder(embedding)
       embed = torch.reshape(embed,(-1,8,4,4))
       out = torch.cat((out,embed),axis=1)
       out = self.conv5(out)
       out = self.output_layer(out)

       return out

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
    """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # Extract the output of the thirty-fifth layer in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
           parameters.requires_grad = False

        # The preprocessing method of the input data. This is the preprocessing method of the VGG model on the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Standardized operations.
        gen = (gen - self.mean) / self.std
        real = (real - self.mean) / self.std
        # Find the feature map difference between the two images.
        loss = torch.nn.functional.l1_loss(self.feature_extractor(gen), self.feature_extractor(real))
        return loss