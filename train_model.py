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

    # data augmentation: rotate, flip, center crop
    def __getitem__(self, index):
        trans = transforms.Compose([
            transforms.CenterCrop(40),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomAffine(20, translate=(0.05,0.1), scale=None, resample=False, fillcolor=0),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
        x = Image.fromarray(self.data[index])
        x = trans(x)
        return x, self.label[index]

    def __len__(self):
        return len(self.data)
    
    
class Concat_noaug(Dataset):
    def __init__(self, data_set, label_set):
        self.data = data_set
        self.label = label_set

    # data augmentation: rotate, flip, center crop
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

batch_size = 256

def get_dataloader(data_path, label_path, aug):
    x = h5py.File(data_path, 'r')['x']
    y = h5py.File(label_path, 'r')['y']
    if aug:
        data_loader = Concat(x, y)
    else:
        data_loader = Concat_noaug(x, y)
        
    data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)
    
    return data_loader

train_loader = get_dataloader('camelyonpatch_level_2_split_train_x.h5', 'camelyonpatch_level_2_split_train_y.h5', True)
valid_loader = get_dataloader('camelyonpatch_level_2_split_valid_x.h5', 'camelyonpatch_level_2_split_valid_y.h5', False)
test_loader = get_dataloader('camelyonpatch_level_2_split_test_x.h5', 'camelyonpatch_level_2_split_test_y.h5', False)

print('Training samples: ', len(train_loader),flush=True)
print('Validation samples: ', len(valid_loader),flush=True)
print('Testing samples: ', len(test_loader),flush=True)

def valid():
#     model.load_state_dict(torch.load('epoch_3'))
    model.eval()
    correct = 0
    l = 0
    for j, i in enumerate(valid_loader):

        with torch.no_grad():

            train_data = i[0].float().to(device)
        #     bc, crop, c, h, w = train_data.size()
        #     train_data = train_data.view(-1, c, h, w)
            pred = model(train_data)
        #     pred = pred.view(bc, crop, -1).mean(1)
            train_label = i[1].view((i[1].shape[0])).long().to(device)

        #     pred = model(train_data)
            loss = F.cross_entropy(pred, train_label)
            l += loss.item()

            pred = torch.argmax(pred, dim=1)
            correct += pred.eq(train_label.view_as(pred)).sum().item()
    print(correct / (len(valid_loader) * batch_size),flush=True)
    
    print('Validation finished',flush=True)
    return correct / (len(valid_loader) * batch_size)

device = torch.device('cuda')
model = models.resnet18(pretrained=True)
model._modules['fc'] = nn.Linear(in_features=512, out_features=2, bias=True) # 2048 for inception
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params,flush=True)

# model = PcamCNN()
model = model.to(device)

# lr = 0.0001
# optimizer = torch.optim.Adam(model.parameters(),lr=lr)
l = []
correct = []
epochs = 50
best_val = 0    
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=0.0001)
for p in range(epochs):
    model.train()
    
    for j, i in enumerate(train_loader):

        train_data = i[0].float().to(device)
    #     bc, crop, c, h, w = train_data.size()
    #     train_data = train_data.view(-1, c, h, w)
        pred = model(train_data)
    #     pred = pred.view(bc, crop, -1).mean(1)
        train_label = i[1].view((i[1].shape[0])).long().to(device)
        optimizer.zero_grad()
    #     pred = model(train_data)
        loss = F.cross_entropy(pred, train_label)# + 0.5 * F.cross_entropy(pred[1], train_label)
        l.append(loss.item())
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(pred, dim=1)
        pred_correct = pred.eq(train_label.view_as(pred)).sum().item() / len(train_label)
        correct.append(pred_correct)
        if (j+1)%100 == 0:
            print('Current batch: ', j+1, 'Current loss: ', loss.item(), 'Current acc: ', pred_correct,flush=True)
#         print('loss: ', loss.item(), 'acc: ', pred.eq(train_label.view_as(pred)).sum().item() / len(train_label))
#     torch.save(model.state_dict(), 'dense_epoch_' + str(p))
    val_acc = valid()
    if val_acc > best_val:
        torch.save(model.state_dict(), 'resnet_epoch_' + str(p) + str(val_acc))
        print('Found best')
        best_val = val_acc
print('Training finished',flush=True)