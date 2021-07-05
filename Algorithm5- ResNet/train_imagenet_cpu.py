import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
from MODELS.vgg import vgg19
from MODELS.fc import fc

from Net.Resnet50 import ResNet50
from cait_models import cait_S24_224
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from pupil_datasets import Datasets
import torchvision.models as models
from MODELS.model_resnet import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='images',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=0, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, default='1', metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default='BAM')
best_prec1 = 0
con_matrix = [[0, 0, 0, 0] for _ in range(4)]
two_class = 0
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

def main():
    global args, best_prec1,con_matrix
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # create model
    if args.arch == "resnet":
        model = ResidualNet( 'ImageNet', args.depth, 4, args.att_type )
    elif args.arch == "resnet50":
        model = ResNet50(num_classes=4)
    elif args.arch == "cait_S24_224":
        model = cait_S24_224(
            pretrained=False,
            num_classes=4,
            drop_rate=0,
            drop_path_rate=0
        )
        if True:
            checkpoint = torch.load('checkpoints/S24_224.pth', map_location='cpu')

            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            model.load_state_dict(checkpoint_model, strict=False)
    elif args.arch == 'vgg19':
        if two_class:
            model = vgg19(num_classes=4)
        else:
            model = models.vgg19(num_classes=4)
    elif args.arch == 'fc':
        model = fc()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                         momentum=args.momentum,
    #                         weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), args.lr,
    #                         weight_decay=args.weight_decay)
    #model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = torch.nn.DataParallel(model)
    print ("model")
    print (model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint,strict=False)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[117.338/255, 55.523/255, 18.479/255], std=[85.576/255, 42.847/255, 17.877/255])
    train_dataset = Datasets(is_color=True,
                       phase='train',
                       transform=transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                                     transforms.RandomVerticalFlip(),
                                                     transforms.ToTensor(),
                                                     normalize, ]))
    test_dataset = Datasets(is_color=True,
                       phase='test',
                       transform=transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                                     transforms.ToTensor(),
                                                     normalize, ]))
    print("已读取%d张图像" % len(train_dataset))
    print("已读取%d张图像" % len(test_dataset))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=False, sampler=train_sampler,drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False,drop_last=False)
    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('./data/', train=True, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    transforms.Resize(224),
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])),
    #     batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('./data/', train=False, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    transforms.Resize(224),
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])),
    #     batch_size=args.batch_size, shuffle=True)
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch


        train(train_loader, model, criterion, optimizer, epoch)
        for i in con_matrix:
            print(i)
        con_matrix = [[0, 0, 0, 0] for _ in range(4)]


        # evaluate on validation set
        # prec1 = validate(val_loader, model, criterion, epoch)
        prec1 = validate(val_loader, model, criterion, epoch)
        for i in con_matrix:
            print(i)
        con_matrix = [[0, 0, 0, 0] for _ in range(4)]
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.prefix)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
    #for i, (input, y_retinopathyGrade) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = torch.as_tensor(target, dtype=torch.long)
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch {0}: [{1}/{2}] |%s|%.0d%%\t'.format(epoch, i, len(train_loader)) %
                  ('█' * int(20 * i / len(train_loader)) +
                   '░' * (20 - int(20 * i / len(train_loader))),
                   int(100 * i / len(train_loader))), end='')

            print('Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Prec@1 {top1.avg:.3f}'.format(
                   batch_time=batch_time,
                   loss=losses, top1=top1))

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
    # for i, (input, y_retinopathyGrade) in enumerate(val_loader):
        # measure data loading time
        #target = torch.as_tensor(y_retinopathyGrade, dtype=torch.long)
        
        # compute output
        res = [[0, 0, 0, 0] for _ in range(target.size(0))]
        loss = []
        with torch.no_grad():
            for input in inputs:
                output = model(input)
                _, pred = output.data.topk(1, 1, True, True)
                loss.append(criterion(output, target))
                pred = pred.t()
                for r,j in enumerate(pred.data[0]):
                    res[r][int(j.item())]+=1
            pred = torch.IntTensor([res[i].index(max(res[i])) for i in range(len(res))])
            correct = pred.eq(target.expand_as(pred.data))

        ####

        for r, j in zip(target.data, pred.data):
            con_matrix[int(r.item())][int(j.item())] += 1
        correct_k = correct.contiguous().view(-1).float().sum(0)
        res=correct_k.mul_(100.0 / target.size(0))
        pred1=res

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        losses.update(sum(loss)/len(inputs), target.size(0))
        top1.update(pred1.item(), target.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Val Epoch {0}: [{1}/{2}] |%s|%.0d%%\t'.format(epoch, i, len(val_loader)) %
                  ('█' * int(20 * i / len(val_loader)) +
                   '░' * (20 - int(20 * i / len(val_loader))),
                   int(100 * i / len(val_loader))), end='')

            print('Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Prec@1 {top1.avg:.3f}'.format(
                batch_time=batch_time,
                loss=losses, top1=top1))
    
    print(' * Prec@1 {top1.avg:.3f}'
            .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, prefix):
    filename='./checkpoints/%s_checkpoint.pth.tar'%prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%prefix)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ####

    for i,j in zip(target.data,pred.data[0]):
        con_matrix[int(i.item())][int(j.item())] += 1

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mul_accuracy(output, target, topk):
    """Computes the precision@k for the specified values of k"""
    threshold = 0.5

    batch_size = target.size(0)

    pred = torch.gt(output,0.5).float()

    correct = [torch.equal(i,j) for i,j in zip(target,pred)]
    correct_k=list.count(correct,True)
    res = correct_k*(100.0 / batch_size)
    return torch.tensor([res,100])


if __name__ == '__main__':
    main()
