import argparse
from quant.utils import *
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
import timm


parser = argparse.ArgumentParser(description='train code')

group = parser.add_argument_group('Model settings')
group.add_argument('--model', metavar='MODEL', type=str,
                  help='model name')
group.add_argument('--pretrained', action='store_true',
                  help='use pretrained model')
group.add_argument('--checkpoint', metavar='CKPT', type=str,
                  help='checkpoint')
group.add_argument('--num-classes', metavar='NUM', type=int,
                  help='num classes')
group = parser.add_argument_group('Dataset settings')
group.add_argument('--dir', metavar='DIR', type=str,
                  help='data directory, e.g., CIFAR10, CIFAR100, or path, default=CIFAR10')
group.add_argument('--img-size', metavar='SIZE', type=int,
                  help='image size')
group.add_argument('--mean', metavar='MEAN', nargs=3, type=float,
                  help='dataset mean')
group.add_argument('--std', metavar='STD', nargs=3, type=float,
                  help='dataset std')
group.add_argument('--stat', metavar='STAT', type=str,
                  help='dataset stat. e.g., CIFAR10, CIFAR100, IMAGENET')
group = parser.add_argument_group('Train settings')
group.add_argument('-b', '--batch-size', metavar='BATCH_SIZE', type=int,
                  help='batch size')
group = parser.add_argument_group('Quantization settings')
group.add_argument('--quant', action='store_true',
                  help='use quantization')
group.add_argument('--num-bit', metavar='BIT', type=int, default=8,
                  help='the number of bit to use')
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--log-freq', metavar='FREQ', type=int, default=10,
                  help='train/val log frequancy, default=10')
group.add_argument('--seed', metavar='SEED', type=int, default=42,
                  help='random seed, default=42')
group.add_argument('--rank', metavar='RANK', type=int, default=0,
                  help='local rank, default=0')

def test(model, epoch, dataloader, criterion, saver):
    args = parser.parse_args()
    global best_acc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx%args.log_freq == 0 or batch_idx+1 == len(dataloader):
                print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, len(dataloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    if saver and args.save_path:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving to best model...')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            saver.save_checkpoint(epoch, metric=acc)
            best_acc = acc
        if epoch == args.epoch-1:
            print('Saving to last model')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            saver.save_checkpoint(epoch)                

def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timm.utils.random_seed(seed=args.seed, rank=args.rank)

    transform_list = []
    if args.img_size:
        transform_list.append(transforms.Resize(args.img_size, args.img_size))
    transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    if args.stat:
        if args.stat=='CIFAR10':
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        elif args.stat=='CIFAR100':
            transform_list.append(transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)))
        elif args.stat=='IMAGENET':
            transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    else:
        transform_list.append(transforms.Normalize(args.mean, args.std))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        *transform_list,
    ])        

    ## Load dataset and dataloader
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    ## Load model
    model = timm.create_model(
            args.model,
            checkpoint_path=args.checkpoint,
            num_classes=args.num_classes,
            pretrained=args.pretrained
            )
    print(f'Model {timm.models.safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
    #  model = resnet50()
    #  load_model(model, args.checkpoint, args.num_classes)
    model.to(device)
    # summary(model, (1,3,args.img_size,args.img_size))
    
    # quantize model
    if args.quant:
        print(f'Symmetric quantization: {args.num_bit}-bit')
        quant_conv_weight(model, kbit=args.num_bit)

    ## Set loss func and optim
    criterion = nn.CrossEntropyLoss()
    
    test(model=model, epoch=None, dataloader=test_dataloader, criterion=criterion, saver=None)

if __name__=='__main__':
    main()
    # for imgs, labels in test_dataloader:
    #     batch_size = imgs[0]
        
    #     out = model(imgs)
        
    #     for idx, (img, pred) in enumerate(zip(imgs, out)):
    #         plt.subplot(2, 4, idx+1)
    #         plt.imshow(img.permute(1,2,0))
    #         plt.title(pred.argmax())
    #     plt.show()
            
    
    
