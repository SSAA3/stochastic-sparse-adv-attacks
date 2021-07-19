"""
Load Dataset and model to use for generating adversarial samples
"""

import torch, torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from data.train.ResNet18 import ResNet18
from data.train.VGG19 import VGG
        
class Load_DataSet:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        
    def param_dataset(self, dataset):
        
        if dataset == 'ImageNet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            max_iter = 5000
            nb_classes = 1000
        
        elif dataset == 'CIFAR10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]

            max_iter = 32*32*3
            nb_classes = 10
        
        else:
            raise ValueError('Invalid dataset argument')

        return mean, std, max_iter, nb_classes


    def transform_data(self, dataset, NN):
        
        if dataset == 'ImageNet':
            
            if NN == 'inception_v3':
                resize = 299
                centercrop = 299

            elif NN == 'vgg16':
                resize = 256
                centercrop = 224

            else:
                raise ValueError('Invalid model argument')
        
        elif dataset == 'CIFAR10':
            resize = centercrop = 32
        
        else:
            raise ValueError('Invalid dataset argument')
         
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(centercrop),
            transforms.ToTensor()])

        return transform
    
    
    def dataset_loader(self, dataset, NN):
        
        if dataset == 'ImageNet':
            Dataset = torchvision.datasets.ImageFolder(root='./data/data_ImageNet', transform=self.transform_data(dataset, NN))
            Dataloader = torch.utils.data.DataLoader(Dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)
            
        elif dataset == 'CIFAR10':
            Dataset = torchvision.datasets.CIFAR10(root='./data/data_CIFAR10', train=False, transform=self.transform_data(dataset, NN))
            Dataloader = torch.utils.data.DataLoader(Dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        
        else:
            raise ValueError('Invalid dataset argument')

        return Dataloader
    
    
    def load_model(self, dataset, NN, job):
        
        if dataset == 'ImageNet':
                
            if NN == 'inception_v3':
                model = models.inception_v3(pretrained=True).train(False)

            elif NN == 'vgg16':
                model = models.vgg16(pretrained=True).train(False)

            else:
                raise ValueError('Invalid model argument')
        
        
        elif dataset == 'CIFAR10':
            
            if NN == 'resnet18':
                
                load_path = torch.load('./data/models/ResNet18_CIFAR10.pt', map_location=torch.device(self.device))
                
                model = ResNet18()
                model = torch.nn.DataParallel(model)
                model.load_state_dict(load_path, strict=False)
                model = model.eval()
                
                
            elif NN == 'vgg19':
                
                load_path = torch.load('./data/models/VGG19_CIFAR10.pt', map_location=torch.device(self.device))
                
                model = VGG('VGG19')
                model = torch.nn.DataParallel(model)
                model.load_state_dict(load_path, strict=False)
                model = model.eval()
            
            
            else:
                raise ValueError('Invalid model argument')
        
        else:
            raise ValueError('Invalid dataset argument')
                
        return model