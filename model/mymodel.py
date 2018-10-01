import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F
import myinceptionv3

class MyModel(nn.Module):

    def __init__(self):

        num_classes = 2
        lstm_hidden_state_dim = 50

        super(MyModel, self).__init__()
        self.cnn = myinceptionv3.my_inception_v3(pretrained=True, aux_logits=False)

        # Create the linear layers that will process both the img and the txt
        # ARCH-1
        # self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        # self.fc2 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, num_classes)

        # ARCH-2 6fc
        # self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        # self.fc2 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048)
        # self.fc3 = nn.Linear(2048, 1024)
        # self.fc4 = nn.Linear(1024, 512)
        # self.fc5 = nn.Linear(512, 512)
        # self.fc6 = nn.Linear(512, num_classes)

        # ARCH-3 8fc
        # self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        # self.fc2 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048)
        # self.fc3 = nn.Linear(2048, 2048)
        # self.fc4 = nn.Linear(2048, 1024)
        # self.fc5 = nn.Linear(1024, 1024)
        # self.fc6 = nn.Linear(1024, 512)
        # self.fc7 = nn.Linear(512, 512)
        # self.fc8 = nn.Linear(512, num_classes)


    def forward(self, image, img_text, tweet):
        x1 = self.cnn(image) # * 0
        x2 = img_text # * 0
        x3 = tweet # * 0

        x = torch.cat((x2, x3), dim=1)
        x = torch.cat((x1, x), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x