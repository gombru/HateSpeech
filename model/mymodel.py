import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import myinceptionv3

class MyModel(nn.Module):

    def __init__(self):

        super(MyModel, self).__init__()
        self.cnn = myinceptionv3.my_inception_v3(pretrained=True, aux_logits=False)
        self.mm = MultiModalNetSpacialConcat()

    def forward(self, image, img_text, tweet):

        x1 = self.cnn(image) # * 0 # CNN
        x2 = img_text # * 0  # Img Text Input
        x3 = tweet # * 0   # Tweet Text Input
        x = self.mm(x1, x2, x3)  # Multimodal net

        return x


class MultiModalNetConcat(nn.Module):

    def __init__(self):
        super(MultiModalNetConcat, self).__init__()
        # Create the linear layers that will process both the img and the txt
        num_classes = 2
        lstm_hidden_state_dim = 50

        # ARCH-1 4fc same dimensions
        # self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, num_classes)

        # ARCH-1 4fc same dimensions
        # Unimodal
        self.cnn_fc1 = nn.Linear(2048, 1024)
        self.img_text_fc1 = nn.Linear(50, 1024)
        self.tweet_text_fc1 = nn.Linear(50, 1024)
        # Multimodal
        self.fc1 = nn.Linear(1024*3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_classes)

        # ARCH-2 6fc
        # self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        # self.fc2 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048)
        # self.fc3 = nn.Linear(2048, 1024)
        # self.fc4 = nn.Linear(1024, 512)
        # self.fc5 = nn.Linear(512, 512)
        # self.fc6 = nn.Linear(512, num_classes)

    def forward(self, x1, x2, x3):

        # Separate process
        x1 = self.cnn_fc1(x1)
        x2 = self.img_text_fc1(x2)
        x3 = self.tweet_text_fc1(x3)

        # Concatenate
        x = torch.cat((x2, x3), dim=1)
        x = torch.cat((x1, x), dim=1)

        # ARCH-1 4fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # ARCH-2 6fc
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = self.fc6(x)

        return x

class MultiModalNetSpacialConcat(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self):
        super(MultiModalNetSpacialConcat, self).__init__()
        # Create the linear layers that will process both the img and the txt
        num_classes = 2
        lstm_hidden_state_dim = 50

        self.MM_InceptionE_1 = InceptionE(2148)
        self.MM_InceptionE_2 = InceptionE(2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x1, x2, x3):
        # Repeat text embeddings in the 8x8 grid
        x2 = x2.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8
        x3 = x3.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x = torch.cat((x2, x3), dim=1) # 100 x 8 x 8
        x = torch.cat((x1, x), dim=1) # 2148 x 8 x 8

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(x) # 2148 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = F.relu(self.fc1(x))  # 1024
        x = F.relu(self.fc2(x)) # 512
        x = F.relu(self.fc3(x))# 2

        return x


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)