import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=7,stride=1,padding=0),#((in_shape-kernel_size)/stride)+1,#294
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),#((in_shape-kernel_size)/stride)+1,146
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=13, stride=1, padding=0),#134
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2,padding=1),#66
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=17, stride=1, padding=0),#50
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2)#24
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding=0),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(800, 128, bias=True),
            # nn.ReLU(),
            # nn.Linear(128, 5, bias=True),
        )
        # self.conv_1 = nn.Conv1d(in_channels=1,out_channels=4,kernel_size=21,stride=1,padding=0)#((in_shape-kernel_size)/stride)+1,300->280
        # self.pool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # ((in_shape-kernel_size)/stride)+1,140
        # self.F = nn.Flatten()
    def forward(self,X):
        out = self.net(X)
        # out = self.conv_1(X)
        # out = self.relu(out)
        # out = self.pool_1(out)
        return out