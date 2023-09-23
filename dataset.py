import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

class BuildDataset(Dataset):
    def __init__(self, params_loaded):
        self.device = torch.device(params_loaded['device'])
        self.params_loaded = params_loaded
        self.embeddings = np.load(params_loaded['data']['embeddings_path'])
        self.images = [f for f in os.listdir(params_loaded['data']['imgs_path']) if os.path.isfile(os.path.join(params_loaded['data']['imgs_path'], f))]
        self.preprocess = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def __getitem__(self,index):
        if self.images[index][:-4] == "6835436191821":
            del self.images[index]
            return self.__getitem__(index)
        img_path = self.params_loaded['data']['imgs_path'] + '/' + self.images[index]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        embedding = torch.Tensor(self.embeddings[self.images[index][:-4]])
        return img, embedding, self.images[index]
    
    def __len__(self):
        return len(self.images)

class BuildDataloader(DataLoader):
  def __init__(self,dataset,batch_size,shuffle,num_workers):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.num_workers = num_workers
  
  def collect_fn(self, batch):
    out_batch = {}
    img_list = []
    embedding_list = []
    name_list = []
    for img, embedding, img_nm in batch:
      img_list.append(img)
      embedding_list.append(embedding)  
      name_list.append(img_nm)
    
    out_batch['images'] = torch.stack(img_list,dim=0)
    out_batch['embedding'] = torch.stack(embedding_list,dim=0)
    out_batch['names'] = name_list
    
    return out_batch
  
  def loader(self):
    return DataLoader(self.dataset,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        num_workers=self.num_workers,
                        collate_fn=self.collect_fn)