import torch
import torch.backends.cudnn as cudnn
from resactnet2 import ResActNet
from utils import *

# the same parameter setting in article
EPOCH = 20
LR = 0.01
load_path = './resactnet1.pth'
save_path = './resactnet2.pth'

def main():
    if not torch.cuda.is_available():
        exit(0)

    cudnn.benchmark = True
    cudnn.enabled = True

    train_loader, test_loader = dataset()

    model = ResActNet()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(load_path))

    bnbias = []
    weight = []
    for name, param in model.named_parameters():
        if len(param.shape) == 1 or 'bias' in name:
            bnbias.append(param)
        else:
            weight.append(param)

    '''
    print('Load Previous Model')
    model.load_state_dict(torch.load('./binary_best.pth'))
    '''

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    '''
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0 - step / epochs))
    '''

    optimizer = torch.optim.SGD([
        {'params': bnbias, 'weight_decay': 0., 'lr': LR},
        {'params': weight, 'weight_decay': 5e-4, 'lr': 0},
    ], momentum = 0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH, last_epoch = -1)

    best_accuracy = 0

    print('Start Training First 20 EPOCHs')

    for epoch in range(EPOCH):
        train(model, train_loader, criterion, optimizer, epoch)
        accuracy = validate(model, test_loader, epoch)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            
        scheduler.step()
    
    print('Best Prec@1 %.3f' % (best_accuracy))
    model.load_state_dict(torch.load(save_path))

    optimizer = torch.optim.SGD([
        {'params': bnbias, 'weight_decay': 0., 'lr': LR},
        {'params': weight, 'weight_decay': 5e-4, 'lr': LR},
    ], momentum = 0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 3 * EPOCH, last_epoch = -1)

    best_accuracy = 0

    print('Start Training Next 60 EPOCHs')

    for epoch in range(3 * EPOCH):
        train(model, train_loader, criterion, optimizer, epoch)
        accuracy = validate(model, test_loader, epoch)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            
        scheduler.step()
    
    print('Best Prec@1 %.3f' % (best_accuracy))
    model.load_state_dict(torch.load(save_path))

    optimizer = torch.optim.SGD([
        {'params': bnbias, 'weight_decay': 0., 'lr': LR},
        {'params': weight, 'weight_decay': 5e-4, 'lr': 0},
    ], momentum = 0.9, nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH, last_epoch = -1)

    best_accuracy = 0

    print('Start Training Last 20 EPOCHs')

    for epoch in range(EPOCH):
        train(model, train_loader, criterion, optimizer, epoch)
        accuracy = validate(model, test_loader, epoch)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            
        scheduler.step()
    
    print('Best Prec@1 %.3f' % (best_accuracy))

if __name__ == '__main__':
    main()
