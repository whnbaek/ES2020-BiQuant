import torch
import torch.backends.cudnn as cudnn
from reactnet1 import ReActNet
from utils import *

# the same parameter setting in article
EPOCH = 100
LR = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
save_path = './reactnet1.pth'

def main():
    if not torch.cuda.is_available():
        exit(0)

    cudnn.benchmark = True
    cudnn.enabled = True

    train_loader, test_loader = dataset()

    model = ReActNet()
    model = torch.nn.DataParallel(model).cuda()

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
    
    optimizer = torch.optim.Adam(
            [{'params' : bnbias},
            {'params' : weight, 'weight_decay' : WEIGHT_DECAY}],
            lr = LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0 - step / EPOCH))
    
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = MOMENTUM,
                                weight_decay = WEIGHT_DECAY, nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH)
    '''
    best_accuracy = 0

    print('Start Training 100 EPOCHs')

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
