from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import torch


def load_imagenette(BATCH_SIZE: int=16, dir_name: str='./data/', download=False):
    
    if download:
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        with tarfile.open(dataset_name + ".tgz", 'r:gz') as tar:
            tar.extractall(path=dir_name)
    
    dataset_name = "imagenette2-320"
    data_dir = dir_name + dataset_name
    
    transform_train = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(),
                                 T.RandomRotation(45), T.ToTensor(),
                                 T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    transform_test = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    trainset = datasets.ImageFolder(data_dir+'/train', transform=transform_train)
    classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 
                        'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    trainset.classes = classes

    val_ds = datasets.ImageFolder(data_dir+'/val', transform=transform_test)
    val_ds.classes = classes

    len_test, len_val = 1000, len(val_ds)-1000
    
    test_dataset, valid_dataset = random_split(val_ds, [len_test, len_val])
    
    test_dataset.classes = classes
    valid_dataset.classes = classes

    train_loader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=1)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    
    return train_loader, val_loader, test_loader, test_dataset
