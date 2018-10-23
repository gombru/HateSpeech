import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import myinceptionv3
import math

class MyModel(nn.Module):

    def __init__(self, gpu=0):

        super(MyModel, self).__init__()
        self.cnn = myinceptionv3.my_inception_v3(pretrained=True, aux_logits=False)
        self.mm = MultiModalNetTextualKernels_v2_NoVisual_NoTextual_ComplexKernels(gpu)
        self.initialize_weights()

    def forward(self, image, img_text, tweet):

        x1 = self.cnn(image) # * 0 # CNN
        x2 = img_text # * 0  # Img Text Input
        x3 = tweet # * 0   # Tweet Text Input
        x = self.mm(x1, x2, x3)  # Multimodal net
        return x

    def initialize_weights(self):
        for m in self.mm.modules(): # Initialize only mm weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MultiModalNetConcat(nn.Module):

    def __init__(self,gpu):
        super(MultiModalNetConcat, self).__init__()

        self.num_classes = 2
        self.lstm_hidden_state_dim = 50

        # ARCH-1 4fc
        # self.fc1 = BasicFC(2048 +  self.lstm_hidden_state_dim * 2, 2048 + self.lstm_hidden_state_dim * 2)
        # self.fc2 = BasicFC(2048, 1024)
        # self.fc3 = BasicFC(1024, 512)
        # self.fc4 = nn.Linear(512, num_classes)

        # ARCH-1 4fc same dimensions
        # Unimodal
        self.cnn_fc1 = BasicFC(2048, 1024)
        self.img_text_fc1 = BasicFC(50, 1024)
        self.tweet_text_fc1 = BasicFC(50, 1024)

        # Multimodal
        self.fc1 = BasicFC(1024*3, 2048)
        self.fc2 = BasicFC(2048, 1024)
        self.fc3 = BasicFC(1024, 512)
        self.fc4 = nn.Linear(512, self.num_classes)

    def forward(self, x1, x2, x3):

        # Separate process
        x1 = self.cnn_fc1(x1)
        x2 = self.img_text_fc1(x2)
        x3 = self.tweet_text_fc1(x3)

        # Concatenate
        x = torch.cat((x2, x3), dim=1)
        x = torch.cat((x1, x), dim=1)

        # ARCH-1 4fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


class MultiModalNetSpacialConcat(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self,gpu):
        super(MultiModalNetSpacialConcat, self).__init__()

        self.num_classes = 2
        self.lstm_hidden_state_dim = 50

        # Create the linear layers that will process both the img and the txt
        self.MM_InceptionE_1 = InceptionE(2148)
        self.MM_InceptionE_2 = InceptionE(2048)
        self.fc1_sc = BasicFC(2048, 1024)
        self.fc2_sc = BasicFC(1024, 512)
        self.fc3_sc = nn.Linear(512, self.num_classes)


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
        x = self.fc1_sc(x)  # 1024
        x = self.fc2_sc(x) # 512
        x = self.fc3_sc(x)# 2

        return x


class MultiModalNetSpacialConcatSameDim(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self,gpu):
        super(MultiModalNetSpacialConcatSameDim, self).__init__()

        self.num_classes = 2
        self.lstm_hidden_state_dim = 50

        # Unimodal
        self.img_text_fc1_sc = BasicFC(50, 2048)
        self.tweet_text_fc1_sc = BasicFC(50, 2048)

        self.MM_InceptionE_1 = InceptionE(2048*3)
        self.MM_InceptionE_2 = InceptionE(2048)
        self.fc1 = BasicFC(2048, 1024)
        self.fc2 = BasicFC(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)


    def forward(self, x1, x2, x3):

        # Separate process
        x2 = self.img_text_fc1_sc(x2)
        x3 = self.tweet_text_fc1_sc(x3)

        # Repeat text embeddings in the 8x8 grid
        x2 = x2.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 1024 x 8 x 8
        x3 = x3.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 1024 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x = torch.cat((x2, x3), dim=1) # 2048 x 8 x 8
        x = torch.cat((x1, x), dim=1) # 3072 x 8 x 8

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(x) # 3072 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc1(x) # 1024
        x = self.fc2(x) # 512
        x = self.fc3(x)# 2

        return x

class MultiModalNetTextualKernels(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, gpu):
        super(MultiModalNetTextualKernels, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_classes = 2
        self.lstm_hidden_state_dim = 50
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = gpu

        # Textual kernels
        self.fc_tweetTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k6 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k7 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k8 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k9 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k10 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)

        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(2048 + self.num_tweetTxt_kernels + self.num_imgTxt_kernels + 100)
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1 = BasicFC(2048, 1024)
        self.fc2 = BasicFC(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)


    def forward(self, x1, x2, x3):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc_tweetTxt_k1(x3)
        tweetTxt_k2 = self.fc_tweetTxt_k2(x3)
        tweetTxt_k3 = self.fc_tweetTxt_k3(x3)
        tweetTxt_k4 = self.fc_tweetTxt_k4(x3)
        tweetTxt_k5 = self.fc_tweetTxt_k5(x3)
        tweetTxt_k6 = self.fc_tweetTxt_k6(x3)
        tweetTxt_k7 = self.fc_tweetTxt_k7(x3)
        tweetTxt_k8 = self.fc_tweetTxt_k8(x3)
        tweetTxt_k9 = self.fc_tweetTxt_k9(x3)
        tweetTxt_k10 = self.fc_tweetTxt_k10(x3)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc_imgTxt_k1(x2)
        imgTxt_k2 = self.fc_imgTxt_k2(x2)
        imgTxt_k3 = self.fc_imgTxt_k3(x2)
        imgTxt_k4 = self.fc_imgTxt_k4(x2)
        imgTxt_k5 = self.fc_imgTxt_k5(x2)

        # Concatenate textual kernels (along 0 dimension)
        tweetTxt_k1 = tweetTxt_k1.unsqueeze(0) # 1 x 2048
        tweetTxt_k2 = tweetTxt_k2.unsqueeze(0)
        tweetTxt_k3 = tweetTxt_k3.unsqueeze(0)
        tweetTxt_k4 = tweetTxt_k4.unsqueeze(0)
        tweetTxt_k5 = tweetTxt_k5.unsqueeze(0)
        tweetTxt_k6 = tweetTxt_k6.unsqueeze(0)
        tweetTxt_k7 = tweetTxt_k7.unsqueeze(0)
        tweetTxt_k8 = tweetTxt_k8.unsqueeze(0)
        tweetTxt_k9 = tweetTxt_k9.unsqueeze(0)
        tweetTxt_k10 = tweetTxt_k10.unsqueeze(0)
        imgTxt_k1 = imgTxt_k1.unsqueeze(0)
        imgTxt_k2 = imgTxt_k2.unsqueeze(0)
        imgTxt_k3 = imgTxt_k3.unsqueeze(0)
        imgTxt_k4 = imgTxt_k4.unsqueeze(0)
        imgTxt_k5 = imgTxt_k5.unsqueeze(0)

        textual_kernels = torch.cat((tweetTxt_k1, tweetTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k5), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k6), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k7), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k8), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k9), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k10), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k1), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k5), dim=0)  # num_tweetTxt_kernels + num_imgTxt_kernels x 2048
        textual_kernels = textual_kernels.unsqueeze(3)
        textual_kernels = textual_kernels.unsqueeze(4)

        batch_size = int(x2.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(x1[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
            #F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Batch normalization and ReLU
        mm_info = F.relu(self.bn_mm_info(mm_info), inplace=True)


        # Concatenate visual feature map with resulting mm info
        x = torch.cat((x1, mm_info), dim=1)  # 2048+K x 8 x 8

        # Repeat text embeddings in the 8x8 grid
        x2 = x2.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8
        x3 = x3.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x23 = torch.cat((x2, x3), dim=1) # 100 x 8 x 8
        x = torch.cat((x, x23), dim=1) # 2048+K+100 x 8 x 8

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(x) # 2048+K+100 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc1(x) # 1024
        x = self.fc2(x) # 512
        x = self.fc3(x) # 2

        return x

class MultiModalNetTextualKernels_NoVisual(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, gpu):
        super(MultiModalNetTextualKernels_NoVisual, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_classes = 2
        self.lstm_hidden_state_dim = 50
        self.num_tweetTxt_kernels = 20 #10
        self.num_imgTxt_kernels = 10 #5
        self.gpu = gpu

        # Textual kernels
        self.fc_tweetTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k6 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k7 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k8 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k9 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k10 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k11 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k12 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k13 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k14 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k15 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k16 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k17 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k18 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k19 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k20 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k6 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k7 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k8 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k9 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k10 = BasicFC(self.lstm_hidden_state_dim, 2048)

        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(self.num_tweetTxt_kernels + self.num_imgTxt_kernels + 100)
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, self.num_classes)


    def forward(self, x1, x2, x3):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc_tweetTxt_k1(x3)
        tweetTxt_k2 = self.fc_tweetTxt_k2(x3)
        tweetTxt_k3 = self.fc_tweetTxt_k3(x3)
        tweetTxt_k4 = self.fc_tweetTxt_k4(x3)
        tweetTxt_k5 = self.fc_tweetTxt_k5(x3)
        tweetTxt_k6 = self.fc_tweetTxt_k6(x3)
        tweetTxt_k7 = self.fc_tweetTxt_k7(x3)
        tweetTxt_k8 = self.fc_tweetTxt_k8(x3)
        tweetTxt_k9 = self.fc_tweetTxt_k9(x3)
        tweetTxt_k10 = self.fc_tweetTxt_k10(x3)
        tweetTxt_k11 = self.fc_tweetTxt_k11(x3)
        tweetTxt_k12 = self.fc_tweetTxt_k12(x3)
        tweetTxt_k13 = self.fc_tweetTxt_k13(x3)
        tweetTxt_k14 = self.fc_tweetTxt_k14(x3)
        tweetTxt_k15 = self.fc_tweetTxt_k15(x3)
        tweetTxt_k16 = self.fc_tweetTxt_k16(x3)
        tweetTxt_k17 = self.fc_tweetTxt_k17(x3)
        tweetTxt_k18 = self.fc_tweetTxt_k18(x3)
        tweetTxt_k19 = self.fc_tweetTxt_k19(x3)
        tweetTxt_k20 = self.fc_tweetTxt_k20(x3)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc_imgTxt_k1(x2)
        imgTxt_k2 = self.fc_imgTxt_k2(x2)
        imgTxt_k3 = self.fc_imgTxt_k3(x2)
        imgTxt_k4 = self.fc_imgTxt_k4(x2)
        imgTxt_k5 = self.fc_imgTxt_k5(x2)
        imgTxt_k6 = self.fc_imgTxt_k1(x2)
        imgTxt_k7 = self.fc_imgTxt_k2(x2)
        imgTxt_k8 = self.fc_imgTxt_k3(x2)
        imgTxt_k9 = self.fc_imgTxt_k4(x2)
        imgTxt_k10 = self.fc_imgTxt_k5(x2)

        # Concatenate textual kernels (along 0 dimension)
        tweetTxt_k1 = tweetTxt_k1.unsqueeze(0) # 1 x 2048
        tweetTxt_k2 = tweetTxt_k2.unsqueeze(0)
        tweetTxt_k3 = tweetTxt_k3.unsqueeze(0)
        tweetTxt_k4 = tweetTxt_k4.unsqueeze(0)
        tweetTxt_k5 = tweetTxt_k5.unsqueeze(0)
        tweetTxt_k6 = tweetTxt_k6.unsqueeze(0)
        tweetTxt_k7 = tweetTxt_k7.unsqueeze(0)
        tweetTxt_k8 = tweetTxt_k8.unsqueeze(0)
        tweetTxt_k9 = tweetTxt_k9.unsqueeze(0)
        tweetTxt_k10 = tweetTxt_k10.unsqueeze(0)
        tweetTxt_k11 = tweetTxt_k11.unsqueeze(0) # 1 x 2048
        tweetTxt_k12 = tweetTxt_k12.unsqueeze(0)
        tweetTxt_k13 = tweetTxt_k13.unsqueeze(0)
        tweetTxt_k14 = tweetTxt_k14.unsqueeze(0)
        tweetTxt_k15 = tweetTxt_k15.unsqueeze(0)
        tweetTxt_k16 = tweetTxt_k16.unsqueeze(0)
        tweetTxt_k17 = tweetTxt_k17.unsqueeze(0)
        tweetTxt_k18 = tweetTxt_k18.unsqueeze(0)
        tweetTxt_k19 = tweetTxt_k19.unsqueeze(0)
        tweetTxt_k20 = tweetTxt_k20.unsqueeze(0)
        imgTxt_k1 = imgTxt_k1.unsqueeze(0)
        imgTxt_k2 = imgTxt_k2.unsqueeze(0)
        imgTxt_k3 = imgTxt_k3.unsqueeze(0)
        imgTxt_k4 = imgTxt_k4.unsqueeze(0)
        imgTxt_k5 = imgTxt_k5.unsqueeze(0)
        imgTxt_k6 = imgTxt_k6.unsqueeze(0)
        imgTxt_k7 = imgTxt_k7.unsqueeze(0)
        imgTxt_k8 = imgTxt_k8.unsqueeze(0)
        imgTxt_k9 = imgTxt_k9.unsqueeze(0)
        imgTxt_k10 = imgTxt_k10.unsqueeze(0)

        textual_kernels = torch.cat((tweetTxt_k1, tweetTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k5), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k6), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k7), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k8), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k9), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k10), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k11), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k12), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k13), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k14), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k15), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k16), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k17), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k18), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k19), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k20), dim=0)

        textual_kernels = torch.cat((textual_kernels, imgTxt_k1), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k5), dim=0)  # num_tweetTxt_kernels + num_imgTxt_kernels x 2048
        textual_kernels = torch.cat((textual_kernels, imgTxt_k6), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k7), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k8), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k9), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k10), dim=0)
        textual_kernels = textual_kernels.unsqueeze(3)
        textual_kernels = textual_kernels.unsqueeze(4)

        batch_size = int(x2.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(x1[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
            #F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Batch normalization and ReLU
        mm_info = F.relu(self.bn_mm_info(mm_info), inplace=True)

        # Repeat text embeddings in the 8x8 grid
        x2 = x2.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8
        x3 = x3.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x23 = torch.cat((x2, x3), dim=1) # 100 x 8 x 8
        x = torch.cat((mm_info, x23), dim=1) # 2048+K+100 x 8 x 8

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(x) # 2048+K+100 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc1_mm(x) # 1024
        x = self.fc2_mm(x) # 512
        x = self.fc3_mm(x) # 2

        return x


class MultiModalNetTextualKernels_NoVisual_NoTextual(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, gpu):
        super(MultiModalNetTextualKernels_NoVisual_NoTextual, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_classes = 2
        self.lstm_hidden_state_dim = 50
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = gpu

        # Textual kernels
        self.fc_tweetTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k6 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k7 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k8 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k9 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_tweetTxt_k10 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc_imgTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)

        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(self.num_tweetTxt_kernels + self.num_imgTxt_kernels)
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, self.num_classes)


    def forward(self, x1, x2, x3):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc_tweetTxt_k1(x3)
        tweetTxt_k2 = self.fc_tweetTxt_k2(x3)
        tweetTxt_k3 = self.fc_tweetTxt_k3(x3)
        tweetTxt_k4 = self.fc_tweetTxt_k4(x3)
        tweetTxt_k5 = self.fc_tweetTxt_k5(x3)
        tweetTxt_k6 = self.fc_tweetTxt_k6(x3)
        tweetTxt_k7 = self.fc_tweetTxt_k7(x3)
        tweetTxt_k8 = self.fc_tweetTxt_k8(x3)
        tweetTxt_k9 = self.fc_tweetTxt_k9(x3)
        tweetTxt_k10 = self.fc_tweetTxt_k10(x3)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc_imgTxt_k1(x2)
        imgTxt_k2 = self.fc_imgTxt_k2(x2)
        imgTxt_k3 = self.fc_imgTxt_k3(x2)
        imgTxt_k4 = self.fc_imgTxt_k4(x2)
        imgTxt_k5 = self.fc_imgTxt_k5(x2)

        # Concatenate textual kernels (along 0 dimension)
        tweetTxt_k1 = tweetTxt_k1.unsqueeze(0) # 1 x 2048
        tweetTxt_k2 = tweetTxt_k2.unsqueeze(0)
        tweetTxt_k3 = tweetTxt_k3.unsqueeze(0)
        tweetTxt_k4 = tweetTxt_k4.unsqueeze(0)
        tweetTxt_k5 = tweetTxt_k5.unsqueeze(0)
        tweetTxt_k6 = tweetTxt_k6.unsqueeze(0)
        tweetTxt_k7 = tweetTxt_k7.unsqueeze(0)
        tweetTxt_k8 = tweetTxt_k8.unsqueeze(0)
        tweetTxt_k9 = tweetTxt_k9.unsqueeze(0)
        tweetTxt_k10 = tweetTxt_k10.unsqueeze(0)
        imgTxt_k1 = imgTxt_k1.unsqueeze(0)
        imgTxt_k2 = imgTxt_k2.unsqueeze(0)
        imgTxt_k3 = imgTxt_k3.unsqueeze(0)
        imgTxt_k4 = imgTxt_k4.unsqueeze(0)
        imgTxt_k5 = imgTxt_k5.unsqueeze(0)

        textual_kernels = torch.cat((tweetTxt_k1, tweetTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k5), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k6), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k7), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k8), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k9), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k10), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k1), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k5), dim=0)  # num_tweetTxt_kernels + num_imgTxt_kernels x 2048
        textual_kernels = textual_kernels.unsqueeze(3)
        textual_kernels = textual_kernels.unsqueeze(4)

        batch_size = int(x2.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(x1[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
            #F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Batch normalization and ReLU
        mm_info = F.relu(self.bn_mm_info(mm_info), inplace=True)

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(mm_info) # 2048+K+100 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc1_mm(x) # 1024
        x = self.fc2_mm(x) # 512
        x = self.fc3_mm(x) # 2

        return x



class MultiModalNetTextualKernels_NoVisual_NoTextual_ComplexKernels(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, gpu):
        super(MultiModalNetTextualKernels_NoVisual_NoTextual_ComplexKernels, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_classes = 2
        self.lstm_hidden_state_dim = 50
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = gpu

        # Textual kernels
        self.fc1_tweetTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k6 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k7 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k8 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k9 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_tweetTxt_k10 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_imgTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_imgTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_imgTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_imgTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 2048)
        self.fc1_imgTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 2048)

        # Textual kernels fc2
        self.fc2_tweetTxt_k1 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k2 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k3 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k4 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k5 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k6 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k7 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k8 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k9 = BasicFC(2048, 2048)
        self.fc2_tweetTxt_k10 = BasicFC(2048, 2048)
        self.fc2_imgTxt_k1 = BasicFC(2048, 2048)
        self.fc2_imgTxt_k2 = BasicFC(2048, 2048)
        self.fc2_imgTxt_k3 = BasicFC(2048, 2048)
        self.fc2_imgTxt_k4 = BasicFC(2048, 2048)
        self.fc2_imgTxt_k5 = BasicFC(2048, 2048)


        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(self.num_tweetTxt_kernels + self.num_imgTxt_kernels)
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, self.num_classes)


    def forward(self, x1, x2, x3):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc1_tweetTxt_k1(x3)
        tweetTxt_k2 = self.fc1_tweetTxt_k2(x3)
        tweetTxt_k3 = self.fc1_tweetTxt_k3(x3)
        tweetTxt_k4 = self.fc1_tweetTxt_k4(x3)
        tweetTxt_k5 = self.fc1_tweetTxt_k5(x3)
        tweetTxt_k6 = self.fc1_tweetTxt_k6(x3)
        tweetTxt_k7 = self.fc1_tweetTxt_k7(x3)
        tweetTxt_k8 = self.fc1_tweetTxt_k8(x3)
        tweetTxt_k9 = self.fc1_tweetTxt_k9(x3)
        tweetTxt_k10 = self.fc1_tweetTxt_k10(x3)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc1_imgTxt_k1(x2)
        imgTxt_k2 = self.fc1_imgTxt_k2(x2)
        imgTxt_k3 = self.fc1_imgTxt_k3(x2)
        imgTxt_k4 = self.fc1_imgTxt_k4(x2)
        imgTxt_k5 = self.fc1_imgTxt_k5(x2)

        # Same for fc2 of each kernel
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc2_tweetTxt_k1(tweetTxt_k1)
        tweetTxt_k2 = self.fc2_tweetTxt_k2(tweetTxt_k2)
        tweetTxt_k3 = self.fc2_tweetTxt_k3(tweetTxt_k3)
        tweetTxt_k4 = self.fc2_tweetTxt_k4(tweetTxt_k4)
        tweetTxt_k5 = self.fc2_tweetTxt_k5(tweetTxt_k5)
        tweetTxt_k6 = self.fc2_tweetTxt_k6(tweetTxt_k6)
        tweetTxt_k7 = self.fc2_tweetTxt_k7(tweetTxt_k7)
        tweetTxt_k8 = self.fc2_tweetTxt_k8(tweetTxt_k8)
        tweetTxt_k9 = self.fc2_tweetTxt_k9(tweetTxt_k9)
        tweetTxt_k10 = self.fc2_tweetTxt_k10(tweetTxt_k10)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc2_imgTxt_k1(imgTxt_k1)
        imgTxt_k2 = self.fc2_imgTxt_k2(imgTxt_k2)
        imgTxt_k3 = self.fc2_imgTxt_k3(imgTxt_k3)
        imgTxt_k4 = self.fc2_imgTxt_k4(imgTxt_k4)
        imgTxt_k5 = self.fc2_imgTxt_k5(imgTxt_k5)

        # Concatenate textual kernels (along 0 dimension)
        tweetTxt_k1 = tweetTxt_k1.unsqueeze(0) # 1 x 2048
        tweetTxt_k2 = tweetTxt_k2.unsqueeze(0)
        tweetTxt_k3 = tweetTxt_k3.unsqueeze(0)
        tweetTxt_k4 = tweetTxt_k4.unsqueeze(0)
        tweetTxt_k5 = tweetTxt_k5.unsqueeze(0)
        tweetTxt_k6 = tweetTxt_k6.unsqueeze(0)
        tweetTxt_k7 = tweetTxt_k7.unsqueeze(0)
        tweetTxt_k8 = tweetTxt_k8.unsqueeze(0)
        tweetTxt_k9 = tweetTxt_k9.unsqueeze(0)
        tweetTxt_k10 = tweetTxt_k10.unsqueeze(0)
        imgTxt_k1 = imgTxt_k1.unsqueeze(0)
        imgTxt_k2 = imgTxt_k2.unsqueeze(0)
        imgTxt_k3 = imgTxt_k3.unsqueeze(0)
        imgTxt_k4 = imgTxt_k4.unsqueeze(0)
        imgTxt_k5 = imgTxt_k5.unsqueeze(0)

        textual_kernels = torch.cat((tweetTxt_k1, tweetTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k5), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k6), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k7), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k8), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k9), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k10), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k1), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k5), dim=0)  # num_tweetTxt_kernels + num_imgTxt_kernels x 2048
        textual_kernels = textual_kernels.unsqueeze(3)
        textual_kernels = textual_kernels.unsqueeze(4)

        batch_size = int(x2.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(x1[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
            #F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Batch normalization and ReLU
        mm_info = F.relu(self.bn_mm_info(mm_info), inplace=True)

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(mm_info) # 2048+K+100 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc1_mm(x) # 1024
        x = self.fc2_mm(x) # 512
        x = self.fc3_mm(x) # 2

        return x


class MultiModalNetTextualKernels_v2_NoVisual_NoTextual_ComplexKernels(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, gpu):
        super(MultiModalNetTextualKernels_v2_NoVisual_NoTextual_ComplexKernels, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_classes = 2
        self.lstm_hidden_state_dim = 50
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = gpu

        # Textual kernels
        self.fc1_tweetTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k6 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k7 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k8 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k9 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_tweetTxt_k10 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_imgTxt_k1 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_imgTxt_k2 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_imgTxt_k3 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_imgTxt_k4 = BasicFC(self.lstm_hidden_state_dim, 768)
        self.fc1_imgTxt_k5 = BasicFC(self.lstm_hidden_state_dim, 768)

        # Textual kernels fc2
        self.fc2_tweetTxt_k1 = BasicFC(768, 768)
        self.fc2_tweetTxt_k2 = BasicFC(768, 768)
        self.fc2_tweetTxt_k3 = BasicFC(768, 768)
        self.fc2_tweetTxt_k4 = BasicFC(768, 768)
        self.fc2_tweetTxt_k5 = BasicFC(768, 768)
        self.fc2_tweetTxt_k6 = BasicFC(768, 768)
        self.fc2_tweetTxt_k7 = BasicFC(768, 768)
        self.fc2_tweetTxt_k8 = BasicFC(768, 768)
        self.fc2_tweetTxt_k9 = BasicFC(768, 768)
        self.fc2_tweetTxt_k10 = BasicFC(768, 768)
        self.fc2_imgTxt_k1 = BasicFC(768, 768)
        self.fc2_imgTxt_k2 = BasicFC(768, 768)
        self.fc2_imgTxt_k3 = BasicFC(768, 768)
        self.fc2_imgTxt_k4 = BasicFC(768, 768)
        self.fc2_imgTxt_k5 = BasicFC(768, 768)


        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionD_1 = InceptionD(self.num_tweetTxt_kernels + self.num_imgTxt_kernels)
        self.MM_InceptionE_1 = InceptionE(527)
        self.MM_InceptionE_2 = InceptionE(2048)
        self.fc_mm = nn.Linear(2048, self.num_classes)


    def forward(self, x1, x2, x3):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc1_tweetTxt_k1(x3)
        tweetTxt_k2 = self.fc1_tweetTxt_k2(x3)
        tweetTxt_k3 = self.fc1_tweetTxt_k3(x3)
        tweetTxt_k4 = self.fc1_tweetTxt_k4(x3)
        tweetTxt_k5 = self.fc1_tweetTxt_k5(x3)
        tweetTxt_k6 = self.fc1_tweetTxt_k6(x3)
        tweetTxt_k7 = self.fc1_tweetTxt_k7(x3)
        tweetTxt_k8 = self.fc1_tweetTxt_k8(x3)
        tweetTxt_k9 = self.fc1_tweetTxt_k9(x3)
        tweetTxt_k10 = self.fc1_tweetTxt_k10(x3)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc1_imgTxt_k1(x2)
        imgTxt_k2 = self.fc1_imgTxt_k2(x2)
        imgTxt_k3 = self.fc1_imgTxt_k3(x2)
        imgTxt_k4 = self.fc1_imgTxt_k4(x2)
        imgTxt_k5 = self.fc1_imgTxt_k5(x2)

        # Same for fc2 of each kernel
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc2_tweetTxt_k1(tweetTxt_k1)
        tweetTxt_k2 = self.fc2_tweetTxt_k2(tweetTxt_k2)
        tweetTxt_k3 = self.fc2_tweetTxt_k3(tweetTxt_k3)
        tweetTxt_k4 = self.fc2_tweetTxt_k4(tweetTxt_k4)
        tweetTxt_k5 = self.fc2_tweetTxt_k5(tweetTxt_k5)
        tweetTxt_k6 = self.fc2_tweetTxt_k6(tweetTxt_k6)
        tweetTxt_k7 = self.fc2_tweetTxt_k7(tweetTxt_k7)
        tweetTxt_k8 = self.fc2_tweetTxt_k8(tweetTxt_k8)
        tweetTxt_k9 = self.fc2_tweetTxt_k9(tweetTxt_k9)
        tweetTxt_k10 = self.fc2_tweetTxt_k10(tweetTxt_k10)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc2_imgTxt_k1(imgTxt_k1)
        imgTxt_k2 = self.fc2_imgTxt_k2(imgTxt_k2)
        imgTxt_k3 = self.fc2_imgTxt_k3(imgTxt_k3)
        imgTxt_k4 = self.fc2_imgTxt_k4(imgTxt_k4)
        imgTxt_k5 = self.fc2_imgTxt_k5(imgTxt_k5)

        # Concatenate textual kernels (along 0 dimension)
        tweetTxt_k1 = tweetTxt_k1.unsqueeze(0) # 1 x 2048
        tweetTxt_k2 = tweetTxt_k2.unsqueeze(0)
        tweetTxt_k3 = tweetTxt_k3.unsqueeze(0)
        tweetTxt_k4 = tweetTxt_k4.unsqueeze(0)
        tweetTxt_k5 = tweetTxt_k5.unsqueeze(0)
        tweetTxt_k6 = tweetTxt_k6.unsqueeze(0)
        tweetTxt_k7 = tweetTxt_k7.unsqueeze(0)
        tweetTxt_k8 = tweetTxt_k8.unsqueeze(0)
        tweetTxt_k9 = tweetTxt_k9.unsqueeze(0)
        tweetTxt_k10 = tweetTxt_k10.unsqueeze(0)
        imgTxt_k1 = imgTxt_k1.unsqueeze(0)
        imgTxt_k2 = imgTxt_k2.unsqueeze(0)
        imgTxt_k3 = imgTxt_k3.unsqueeze(0)
        imgTxt_k4 = imgTxt_k4.unsqueeze(0)
        imgTxt_k5 = imgTxt_k5.unsqueeze(0)

        textual_kernels = torch.cat((tweetTxt_k1, tweetTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k5), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k6), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k7), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k8), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k9), dim=0)
        textual_kernels = torch.cat((textual_kernels, tweetTxt_k10), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k1), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k2), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k3), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k4), dim=0)
        textual_kernels = torch.cat((textual_kernels, imgTxt_k5), dim=0)  # num_tweetTxt_kernels + num_imgTxt_kernels x 2048
        textual_kernels = textual_kernels.unsqueeze(3)
        textual_kernels = textual_kernels.unsqueeze(4)

        batch_size = int(x2.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,17,17).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(x1[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
            #F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Batch normalization and ReLU
        mm_info = F.relu(self.bn_mm_info(mm_info), inplace=True)

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionD_1(mm_info)
        x = self.MM_InceptionE_1(x) # 2048+K+100 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc_mm(x) # 2

        return x


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

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

class BasicFC(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)