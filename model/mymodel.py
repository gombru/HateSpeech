import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self):

        num_classes = 2
        lstm_hidden_state_dim = 50

        super(MyModel, self).__init__()
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)

        # Delete last fc that maps 2048 features to 1000 classes.
        # Now the output of CNN is the 2048 features
        del(self.cnn._modules['fc']) # Had to remove manually fc from forward pass at inception.py

        # Create the linear layers that will process both the img and the txt
        self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        self.fc2 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_classes)


    def forward(self, image, img_text, tweet):
        x1 = self.cnn(image) # * 0
        x2 = img_text # * 0
        x3 = tweet # * 0

        x = torch.cat((x2, x3), dim=1)
        x = torch.cat((x1, x), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x