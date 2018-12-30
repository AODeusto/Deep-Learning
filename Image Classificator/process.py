from torchvision import datasets, transforms, models
import torch
import os
from PIL import Image
import numpy as np

def process_image_train(data_dir):
    data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])]),
                       'valid' : transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test' : transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])  }
    print('Creating Datasets')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = data_transforms[x])
                      for x in ['train', 'valid', 'test'] }
    print('Datasets done -----> Creating Dataloaders')
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 64, shuffle = True)
                      for x in ['train', 'valid', 'test'] }
    print('DataLoaders ready')
    class_to_idx = image_datasets['train'].class_to_idx
    return dataloader, class_to_idx

def process_image_predict(path_to_image):
    img = Image.open(path_to_image)
    # Resize shortest side to 256 pixels
    if img.size[1] < img.size[0]:
        img.thumbnail([10000,256])
    else:
        img.thumbnail([256,10000])
    # Crop center 224x224
    lower = (img.size[1]-224)/2
    left = (img.size[0]-224)/2
    upper = lower + 224
    right = left + 224

    img = img.crop((left, lower, right, upper))
    # Convert img into a np.array and reorder dimensions, placing n_channels first
    # Scale values to a range 0-1
    scaled_np_img = np.array(img) / 255
    # Normalize color channels
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_np_img = (scaled_np_img - means) / std
    # Place number of color channels first, transpose dimensions of array
    img_ready = norm_np_img.transpose((2,0,1))
    return img_ready
