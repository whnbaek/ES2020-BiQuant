import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.nn.modules import loss
import time

data_path = './data/'
batch_size = 128
num_workers = 8
print_freq = 39
test_freq = 39

# copied from utils
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(logits, target):
        _, pred = logits.topk(k = 1, dim = 1)
        pred = pred[ : , 0]
        correct = pred.eq(target).float().sum().mul_(100.0 / batch_size)

        return correct

def dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
    ])

    train_set = torchvision.datasets.CIFAR10(root = data_path, train = True, \
                                             transform = transform_train, download = True)
    test_set = torchvision.datasets.CIFAR10(root = data_path, train = False, \
                                             transform = transform_test, download = True)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, \
                                               num_workers = num_workers, drop_last = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, \
                                              num_workers = num_workers, drop_last = True)
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        # compute output
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1 = accuracy(logits.data, target)
        n = images.size(0)
        losses.update(loss.data.item(), n)
        top1.update(prec1.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((i + 1) % print_freq) == 0:
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch {0}: [{1}/{2}]\t'
                  'Time {batch_time.val:.4f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i + 1, len(train_loader), \
                                batch_time = batch_time, loss = losses, top1 = top1), flush = True)

def validate(model, test_loader, epoch):
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for images, target in test_loader:
        images = images.cuda()
        target = target.cuda()        

        # compute output
        logits = model(images)      

        # record loss and accuracy
        prec1 = accuracy(logits.data, target)
        n = images.size(0)
        top1.update(prec1.item(), n)

    print(' * Prec@1 {top1.avg:.3f}'.format(top1 = top1))
    
    return top1.avg

'''
class DistributionLoss(loss._Loss):
    def forward(self, model_output, real_output):

        model_output_log_prob = F.log_softmax(model_output, dim = 1)
        real_output_soft = F.softmax(real_output, dim = 1)
        del model_output, real_output

        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()

        return cross_entropy_loss
'''
