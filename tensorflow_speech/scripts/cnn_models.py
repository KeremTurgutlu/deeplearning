from imports import *


class cnn_trad_pool2_net(nn.Module):
    def __init__(self, in_shape=(1, 32, 128), num_classes=12):
        super(cnn_trad_pool2_net, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(16, 8), stride=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(8, 4), stride=(1, 1))
        self.fc = nn.Linear(3648, num_classes)

    def forward(self, x):
        # maxpool + relu is faster than relu + maxpool equivalent ops
        # https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216/4
        x = self.conv1_bn(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return x