import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryAct(nn.Module): # float -> binary
    def __init__(self):
        super(BinaryAct, self).__init__()
    
    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad = True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class NormalBlock(nn.Module):
    def __init__(self, channels):
        super(NormalBlock, self).__init__()
        self.act = BinaryAct()

        self.bias1 = LearnableBias(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.bias2 = LearnableBias(channels)
        self.prelu1 = nn.PReLU(channels)
        self.bias3 = LearnableBias(channels)

        self.bias4 = LearnableBias(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.bias5 = LearnableBias(channels)
        self.prelu2 = nn.PReLU(channels)
        self.bias6 = LearnableBias(channels)
    
    def forward(self, x):
        out1 = self.bn1(self.conv1(self.bias1(x)))
        out1 = x + out1
        out1 = self.bias3(self.prelu1(self.bias2(out1)))

        out2 = self.bn2(self.conv2(self.bias4(out1)))
        out2 = out1 + out2
        out2 = self.bias6(self.prelu1(self.bias5(out2)))

        return out2

class ReductionBlock(nn.Module):
    def __init__(self, channels, stride = 2):
        super(ReductionBlock, self).__init__()
        self.act = BinaryAct()

        self.bias1 = LearnableBias(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, stride = stride, padding = 1, \
                               bias = False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.bias2 = LearnableBias(channels)
        self.prelu1 = nn.PReLU(channels)
        self.bias3 = LearnableBias(channels)

        self.bias4 = LearnableBias(channels)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size = 1, bias = False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size = 1, bias = False)
        self.bn2_1 = nn.BatchNorm2d(channels)
        self.bn2_2 = nn.BatchNorm2d(channels)

        self.bias5 = LearnableBias(2 * channels)
        self.prelu2 = nn.PReLU(2 * channels)
        self.bias6 = LearnableBias(2 * channels)

        self.stride = stride
        
    def forward(self, x):
        out1 = self.bn1(self.conv1(self.bias1(x)))
        if self.stride == 2:
            x = self.pool(x)
        out1 = x + out1
        out1 = self.bias3(self.prelu1(self.bias2(out1)))

        out2 = self.bias4(out1)
        out2_1 = self.bn2_1(self.conv2_1(out2))
        out2_2 = self.bn2_2(self.conv2_2(out2))
        out2_1 = out1 + out2_1
        out2_2 = out1 + out2_2
        out2 = torch.cat([out2_1, out2_2], dim = 1)
        out2 = self.bias6(self.prelu2(self.bias5(out2)))

        return out2

class ReActNet(nn.Module):
    def __init__(self):
        super(ReActNet, self).__init__()
        self.conv = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, \
                              stride = 2, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(num_features = 32)

        self.block1 = ReductionBlock(channels = 32, stride = 1)
        self.block2 = ReductionBlock(channels = 64)
        self.block3 = NormalBlock(channels = 128)
        self.block4 = ReductionBlock(channels = 128)
        self.block5 = NormalBlock(channels = 256)
        self.block6 = ReductionBlock(channels = 256)
        self.block7 = NormalBlock(channels = 512)
        self.block8 = NormalBlock(channels = 512)
        self.block9 = NormalBlock(channels = 512)
        self.block10 = NormalBlock(channels = 512)
        self.block11 = NormalBlock(channels = 512)
        self.block12 = ReductionBlock(channels = 512)
        self.block13 = NormalBlock(channels = 1024)

        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc = nn.Linear(in_features = 1024, out_features = 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
