import torch
import h5py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class Concat(Dataset):
    def __init__(self, data_set, label_set):
        self.data = data_set
        self.label = label_set

    def __getitem__(self, index):
        trans = transforms.Compose([
            transforms.CenterCrop(40),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
        x = Image.fromarray(self.data[index])
        x = trans(x)
        return x, self.label[index]

    def __len__(self):
        return len(self.data)

batch_size = 1
def get_dataloader(data_path, label_path):
    x = h5py.File(data_path, 'r')['x']
    y = h5py.File(label_path, 'r')['y']
    data_loader = Concat(x, y)
    data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)
    
    return data_loader

test_loader = get_dataloader('camelyonpatch_level_2_split_test_x.h5', 'camelyonpatch_level_2_split_test_y.h5')

print('Testing samples: ', len(test_loader),flush=True)


device = torch.device('cpu')
model = models.resnet18(pretrained=True)
model._modules['fc'] = nn.Linear(in_features=512, out_features=2, bias=True)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params,flush=True)

model = model.to(device)
model.load_state_dict(torch.load('resnet_epoch_130.8466796875',map_location='cpu'))


def test():
    model.eval()
    correct = 0
    l = 0
    for j, i in enumerate(test_loader):

        with torch.no_grad():

            train_data = i[0].float().to(device)
            pred = model(train_data)

            train_label = i[1].view((i[1].shape[0])).long().to(device)

            loss = F.cross_entropy(pred, train_label)
            l += loss.item()
            print(pred, i[1])

            pred = torch.argmax(pred, dim=1)
            correct += pred.eq(train_label.view_as(pred)).sum().item()

    print(correct / (len(test_loader) * batch_size),flush=True)
    
    print('Resnet Validation finished',flush=True)
    return correct / (len(test_loader) * batch_size)


test()
