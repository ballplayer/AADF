import torch
from torch import nn
from torch.nn import functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            # 关于此处的inplace，选择是否覆盖，是否使用Relu得到的结果覆盖Relu之前的结果，如果使用inplace进行覆盖
            # 可以节约内存，不需要单独创建变量保存数据

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2),

        )
        self.fc = nn.Sequential(
            # 第一个全连接层的维度不好确定，可以在main函数中创建一个随机张量，输出最后一个卷积层平展开的维度
            # 或者使用nn.AdaptiveMaxPool1d((m,n))直接指定输出的维度
            nn.Linear(in_features=128000, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.layer1(x)  # torch.Size([256, 64, 4000])
        x = self.layer2(x)  # torch.Size([256, 128, 2000])
        x = self.layer3(x)  # torch.Size([256, 256, 1000])
        x = self.layer4(x)  # torch.Size([256, 512, 500])
        x = self.layer5(x)  # torch.Size([256, 512, 250])
        x = x.view(x.size()[0], -1)  # torch.Size([256, 128000])
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = VGG16()
    X = torch.rand(256, 1, 8000)

    # for layer in model:
    #     X = layer(X)
    #     print(layer)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    print(model(X).shape)
