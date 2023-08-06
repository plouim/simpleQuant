from quant import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from torch import optim
import os
from torchvision import transforms
from torchinfo import summary

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        print(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    if SAVE:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..') 
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

SAVE = True
best_acc = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
## Load dataset and dataloader
train_dataset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=ToTensor())
test_dataset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



## Load model
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model = resnet50()
load_model(model, './checkpoint/ckpt.pth', 10)
# model.fc = nn.Linear(model.fc.in_features, 10)
summary(model, (1,3,32,32))

# quantize model
quant_conv_weight(model, kbit=8)

## Set loss func and optim
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, 
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(0, 10):
    # train(epoch)
    test(epoch)
    # scheduler.step()



# for imgs, labels in test_dataloader:
#     batch_size = imgs[0]
    
#     out = model(imgs)
    
#     for idx, (img, pred) in enumerate(zip(imgs, out)):
#         plt.subplot(2, 4, idx+1)
#         plt.imshow(img.permute(1,2,0))
#         plt.title(pred.argmax())
#     plt.show()
        
    
    