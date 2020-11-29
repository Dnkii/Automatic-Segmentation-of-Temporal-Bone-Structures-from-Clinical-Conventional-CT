from torch import nn
import torch

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = in_channels//2 if in_channels > out_channels else out_channels//2
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class wnetDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True, concat=False):
        super(wnetDown, self).__init__()
        if concat==False:
            self.pub = pub(in_channels, out_channels, batch_norm)
        else:
            self.pub = pub(in_channels*3, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x, x1=None):
        x = self.pool(x)
        if x1 is not None:
            x = torch.cat((x, x1), dim=1)
        x = self.pub(x)
        return x

class wnetUp(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(wnetUp, self).__init__()
        self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = torch.cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class wnet(nn.Module):

    def __init__(self, init_channels=1, class_nums=1, batch_norm=True, sample=True):
        super(wnet, self).__init__()
        self.down1 = pub(init_channels, 64, batch_norm)
        self.down2 = wnetDown(64, 128, batch_norm)
        self.down3 = wnetDown(64, 128, batch_norm,concat=True)
        self.up1 = wnetUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.up1(x2, x1)
        x4 = self.down3(x3,x2)
        x = self.up1(x4, x3)
        x = self.con_last(x)
        return self.softmax(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(1, 1, 80, 80, 80).to(device) # 这里的对应前面fforward的输入是32
    net = wnet(1,4,batch_norm=False, sample=False).to(device)
    out = net(inputs)
    netsize=count_param(net)
    print(out.size(),"params:%0.3fM"%(netsize/1000000),"(%s)"%netsize)
    input("按任意键结束")