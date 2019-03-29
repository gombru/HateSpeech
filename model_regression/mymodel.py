import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import myinceptionv3
import math

class MyModel(nn.Module):

    def __init__(self, gpu=0):

        super(MyModel, self).__init__()
        c = {}
        c['num_classes'] = 1
        c['lstm_hidden_state_dim'] = 150
        c['gpu'] = gpu
        self.cnn = myinceptionv3.my_inception_v3(pretrained=True, aux_logits=False)
        self.mm = FCM(c)
        self.initialize_weights()

    def forward(self, image, img_text, tweet):

        i = self.cnn(image) * 0 # CNN
        it = img_text * 0  # Img Text Input
        tt = tweet # * 0   # Tweet Text Input
        x = self.mm(i, it, tt) # Multimodal net
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


class testCNN(nn.Module):

    def __init__(self, c):
        super(testCNN, self).__init__()
        self.fc1 = nn.Linear(2048, c['num_classes'])

    def forward(self, i, it, tt):
        x = self.fc1(i)
        return x


class FCM(nn.Module):

    def __init__(self, c):
        super(FCM, self).__init__()

        # Unimodal
        self.cnn_fc1 = BasicFC(2048, 1024)
        self.img_text_fc1 = BasicFC(c['lstm_hidden_state_dim'], 1024)
        self.tweet_text_fc1 = BasicFC(c['lstm_hidden_state_dim'], 1024)

        # Multimodal
        self.fc1 = BasicFC(1024*3, 2048)
        self.fc2 = BasicFC(2048, 1024)
        self.fc3 = BasicFC(1024, 512)
        self.fc4 = nn.Linear(512, c['num_classes'])

    def forward(self, i, it, tt):

        # tt = F.dropout(tt, p=0.5, training=self.training)
        # it = F.dropout(it, p=0.5, training=self.training)

        # Separate process
        i = self.cnn_fc1(i)
        it = self.img_text_fc1(it)
        tt = self.tweet_text_fc1(tt)

        # Concatenate
        x = torch.cat((it, tt), dim=1)
        x = torch.cat((i, x), dim=1)

        # ARCH-1 4fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

class FCM_tiny(nn.Module):

    def __init__(self, c):
        super(FCM_tiny, self).__init__()

        # Unimodal
        self.cnn_fc1 = BasicFC(2048, 512)
        self.cnn_fc2 = BasicFC(512, c['lstm_hidden_state_dim'] * 2)

        # Multimodal
        self.fc1 = BasicFC(c['lstm_hidden_state_dim'] * 4, c['lstm_hidden_state_dim'] * 4)
        self.fc2 = BasicFC(c['lstm_hidden_state_dim'] * 4, 128)
        self.fc3 = nn.Linear(128, c['num_classes'])

    def forward(self, i, it, tt):

        # tt = F.dropout(tt, p=0.7, training=self.training)
        # it = F.dropout(it, p=0.7, training=self.training)

        # Separate process
        i = self.cnn_fc1(i)
        i = self.cnn_fc2(i)

        # Concatenate
        x = torch.cat((it, tt), dim=1)
        x = torch.cat((i, x), dim=1)

        # MM fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class SCM(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self,c):
        super(SCM, self).__init__()

        # Create the linear layers that will process both the img and the txt
        self.MM_InceptionE_1 = InceptionE(2048 + 2*c['lstm_hidden_state_dim'])
        self.MM_InceptionE_2 = InceptionE(2048)
        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, c['num_classes'])


    def forward(self, i, it, tt):

        # Repeat text embeddings in the 8x8 grid
        it = it.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8
        tt = tt.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x = torch.cat((it, tt), dim=1) # 100 x 8 x 8
        x = torch.cat((i, x), dim=1) # 2148 x 8 x 8

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(x) # 2148 x 8 x 8
        x = self.MM_InceptionE_2(x) # 2048 x 8 x 8

        # AVG Pooling as in Inception
        x = F.avg_pool2d(x, kernel_size=8)  # 2048 x 1 x 1

        # Dropout
        x = F.dropout(x, training=self.training)

        # Reshape and FC layers
        x = x.view(x.size(0), -1) # 2048
        x = self.fc1_mm(x)  # 1024
        x = self.fc2_mm(x)  # 512
        x = self.fc3_mm(x) # 2

        return x


class SCM_SameDim(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self,c):
        super(SCM_SameDim, self).__init__()

        # Unimodal
        self.img_text_fc1_sc = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.tweet_text_fc1_sc = BasicFC(c['lstm_hidden_state_dim'], 2048)

        self.MM_InceptionE_1 = InceptionE(2048*3)
        self.MM_InceptionE_2 = InceptionE(2048)
        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, c['num_classes'])


    def forward(self, i, it, tt):

        # Separate process
        it = self.img_text_fc1_sc(it)
        tt = self.tweet_text_fc1_sc(tt)

        # Repeat text embeddings in the 8x8 grid
        it = it.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 1024 x 8 x 8
        tt = tt.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 1024 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x = torch.cat((it, tt), dim=1) # 2048 x 8 x 8
        x = torch.cat((i, x), dim=1) # 3072 x 8 x 8

        # 1x1 Convolutions using Inceptions E blocks
        x = self.MM_InceptionE_1(x) # 3072 x 8 x 8
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

class TKM_ConcatAll(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, c):
        super(TKM_ConcatAll, self).__init__()
        
        # Create the linear layers that will process both the img and the txt
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = c['gpu']

        # Textual kernels
        self.fc_tweetTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k4 = BasicFC(['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k5 = BasicFC(['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)

        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(2048 + self.num_tweetTxt_kernels + self.num_imgTxt_kernels + 2 * c['lstm_hidden_state_dim'])
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1 = BasicFC(2048, 1024)
        self.fc2 = BasicFC(1024, 512)
        self.fc3 = nn.Linear(512, c['num_classes'])


    def forward(self, i, it, tt):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc_tweetTxt_k1(tt)
        tweetTxt_k2 = self.fc_tweetTxt_k2(tt)
        tweetTxt_k3 = self.fc_tweetTxt_k3(tt)
        tweetTxt_k4 = self.fc_tweetTxt_k4(tt)
        tweetTxt_k5 = self.fc_tweetTxt_k5(tt)
        tweetTxt_k6 = self.fc_tweetTxt_k6(tt)
        tweetTxt_k7 = self.fc_tweetTxt_k7(tt)
        tweetTxt_k8 = self.fc_tweetTxt_k8(tt)
        tweetTxt_k9 = self.fc_tweetTxt_k9(tt)
        tweetTxt_k10 = self.fc_tweetTxt_k10(tt)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc_imgTxt_k1(it)
        imgTxt_k2 = self.fc_imgTxt_k2(it)
        imgTxt_k3 = self.fc_imgTxt_k3(it)
        imgTxt_k4 = self.fc_imgTxt_k4(it)
        imgTxt_k5 = self.fc_imgTxt_k5(it)

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

        batch_size = int(it.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(i[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
            #F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Batch normalization and ReLU
        mm_info = F.relu(self.bn_mm_info(mm_info), inplace=True)


        # Concatenate visual feature map with resulting mm info
        x = torch.cat((i, mm_info), dim=1)  # 2048+K x 8 x 8

        # Repeat text embeddings in the 8x8 grid
        it = it.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8
        tt = tt.unsqueeze(2).unsqueeze(2).repeat(1, 1, 8, 8) # 50 x 8 x 8

        # Concatenate text embeddings in each 8x8 cell
        x23 = torch.cat((it, tt), dim=1) # 100 x 8 x 8
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

class TKM_NoVisualConcat(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, c):
        super(TKM_NoVisualConcat, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_tweetTxt_kernels = 20 # 10
        self.num_imgTxt_kernels = 10 # 5
        self.gpu = c['gpu']

        # Textual kernels
        self.fc_tweetTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k11 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k12 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k13 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k14 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k15 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k16 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k17 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k18 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k19 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k20 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 2048)

        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(self.num_tweetTxt_kernels + self.num_imgTxt_kernels + c['lstm_hidden_state_dim']*2)
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, c['num_classes'])


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
class TKM_NoVisualNoTextualConcat(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, c):
        super(TKM_NoVisualNoTextualConcat, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = c['gpu']

        # Textual kernels
        self.fc_tweetTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_tweetTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc_imgTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)

        self.bn_mm_info = nn.BatchNorm2d(self.num_tweetTxt_kernels + self.num_imgTxt_kernels, eps=0.001)

        self.MM_InceptionE_1 = InceptionE(self.num_tweetTxt_kernels + self.num_imgTxt_kernels)
        self.MM_InceptionE_2 = InceptionE(2048)

        self.fc1_mm = BasicFC(2048, 1024)
        self.fc2_mm = BasicFC(1024, 512)
        self.fc3_mm = nn.Linear(512, c['num_classes'])


    def forward(self, i, it, tt):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc_tweetTxt_k1(tt)
        tweetTxt_k2 = self.fc_tweetTxt_k2(tt)
        tweetTxt_k3 = self.fc_tweetTxt_k3(tt)
        tweetTxt_k4 = self.fc_tweetTxt_k4(tt)
        tweetTxt_k5 = self.fc_tweetTxt_k5(tt)
        tweetTxt_k6 = self.fc_tweetTxt_k6(tt)
        tweetTxt_k7 = self.fc_tweetTxt_k7(tt)
        tweetTxt_k8 = self.fc_tweetTxt_k8(tt)
        tweetTxt_k9 = self.fc_tweetTxt_k9(tt)
        tweetTxt_k10 = self.fc_tweetTxt_k10(tt)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc_imgTxt_k1(it)
        imgTxt_k2 = self.fc_imgTxt_k2(it)
        imgTxt_k3 = self.fc_imgTxt_k3(it)
        imgTxt_k4 = self.fc_imgTxt_k4(it)
        imgTxt_k5 = self.fc_imgTxt_k5(it)

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

        batch_size = int(it.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(i[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
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


class CTKM_NoVisualNoTextualConcat(nn.Module):

    # CNN input size: 8 x 8 x 2048
    def __init__(self, c):
        super(CTKM_NoVisualNoTextualConcat, self).__init__()

        # Create the linear layers that will process both the img and the txt
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = c['gpu']

        # Textual kernels
        self.fc1_tweetTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_tweetTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_imgTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_imgTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_imgTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_imgTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 2048)
        self.fc1_imgTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 2048)

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
        self.fc3_mm = nn.Linear(512, c['num_classes'])


    def forward(self, i, it, tt):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc1_tweetTxt_k1(tt)
        tweetTxt_k2 = self.fc1_tweetTxt_k2(tt)
        tweetTxt_k3 = self.fc1_tweetTxt_k3(tt)
        tweetTxt_k4 = self.fc1_tweetTxt_k4(tt)
        tweetTxt_k5 = self.fc1_tweetTxt_k5(tt)
        tweetTxt_k6 = self.fc1_tweetTxt_k6(tt)
        tweetTxt_k7 = self.fc1_tweetTxt_k7(tt)
        tweetTxt_k8 = self.fc1_tweetTxt_k8(tt)
        tweetTxt_k9 = self.fc1_tweetTxt_k9(tt)
        tweetTxt_k10 = self.fc1_tweetTxt_k10(tt)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc1_imgTxt_k1(it)
        imgTxt_k2 = self.fc1_imgTxt_k2(it)
        imgTxt_k3 = self.fc1_imgTxt_k3(it)
        imgTxt_k4 = self.fc1_imgTxt_k4(it)
        imgTxt_k5 = self.fc1_imgTxt_k5(it)

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

        batch_size = int(it.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,8,8).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(i[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
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


class CTKM_v2_NoVisualNoTextual(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, c):
        super(CTKM_v2_NoVisualNoTextual, self).__init__()
        # Create the linear layers that will process both the img and the txt
        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = c['gpu']

        # Textual kernels
        self.fc1_tweetTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 768)

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
        self.fc_mm = nn.Linear(2048, c['num_classes'])


    def forward(self, i, it, tt):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc1_tweetTxt_k1(tt)
        tweetTxt_k2 = self.fc1_tweetTxt_k2(tt)
        tweetTxt_k3 = self.fc1_tweetTxt_k3(tt)
        tweetTxt_k4 = self.fc1_tweetTxt_k4(tt)
        tweetTxt_k5 = self.fc1_tweetTxt_k5(tt)
        tweetTxt_k6 = self.fc1_tweetTxt_k6(tt)
        tweetTxt_k7 = self.fc1_tweetTxt_k7(tt)
        tweetTxt_k8 = self.fc1_tweetTxt_k8(tt)
        tweetTxt_k9 = self.fc1_tweetTxt_k9(tt)
        tweetTxt_k10 = self.fc1_tweetTxt_k10(tt)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc1_imgTxt_k1(it)
        imgTxt_k2 = self.fc1_imgTxt_k2(it)
        imgTxt_k3 = self.fc1_imgTxt_k3(it)
        imgTxt_k4 = self.fc1_imgTxt_k4(it)
        imgTxt_k5 = self.fc1_imgTxt_k5(it)

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

        batch_size = int(it.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,17,17).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(i[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
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

class CTKM_v2_ConcatAll(nn.Module):
    # CNN input size: 8 x 8 x 2048
    def __init__(self, c):
        super(CTKM_v2_ConcatAll, self).__init__()
        # Create the linear layers that will process both the img and the txt

        self.num_tweetTxt_kernels = 10
        self.num_imgTxt_kernels = 5
        self.gpu = c['gpu']

        # Textual kernels
        self.fc1_tweetTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k6 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k7 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k8 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k9 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_tweetTxt_k10 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k1 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k2 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k3 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k4 = BasicFC(c['lstm_hidden_state_dim'], 768)
        self.fc1_imgTxt_k5 = BasicFC(c['lstm_hidden_state_dim'], 768)

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

        self.fc1_mm = BasicFC(2048+ c['lstm_hidden_state_dim'] * 2, 2048)
        self.fc2_mm = BasicFC(2048, 2048)
        self.fc3_mm = nn.Linear(2048, c['num_classes'])


    def forward(self, i, it, tt):

        # Learn K ((2)10) kernels from Text embeddings
        # Kernels Tweet Text # 2048 x 1 x 1
        tweetTxt_k1 = self.fc1_tweetTxt_k1(tt)
        tweetTxt_k2 = self.fc1_tweetTxt_k2(tt)
        tweetTxt_k3 = self.fc1_tweetTxt_k3(tt)
        tweetTxt_k4 = self.fc1_tweetTxt_k4(tt)
        tweetTxt_k5 = self.fc1_tweetTxt_k5(tt)
        tweetTxt_k6 = self.fc1_tweetTxt_k6(tt)
        tweetTxt_k7 = self.fc1_tweetTxt_k7(tt)
        tweetTxt_k8 = self.fc1_tweetTxt_k8(tt)
        tweetTxt_k9 = self.fc1_tweetTxt_k9(tt)
        tweetTxt_k10 = self.fc1_tweetTxt_k10(tt)
        # Kernels Image Text # 2048 x 1 x 1
        imgTxt_k1 = self.fc1_imgTxt_k1(it)
        imgTxt_k2 = self.fc1_imgTxt_k2(it)
        imgTxt_k3 = self.fc1_imgTxt_k3(it)
        imgTxt_k4 = self.fc1_imgTxt_k4(it)
        imgTxt_k5 = self.fc1_imgTxt_k5(it)

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

        batch_size = int(it.shape[0]) # Batch size can be different in some iters

        # Apply 1x1x2048 kernels to visual feature map
        #     input: input tensor of shape (:math:`minibatch \times in\_channels \times iH \times iW`)
        #     weight: filters of shape (:math:`out\_channels \times \frac{in\_channels}{groups} \times kH \times kW`)
        #   --> But we have different filters for batch element, so we have to do it element by element
        mm_info = torch.cuda.FloatTensor(batch_size,self.num_tweetTxt_kernels+self.num_imgTxt_kernels,17,17).cuda(self.gpu)
        #m_info[batch_size,k,8,8]
        for batch_i in range(0,batch_size):
            mm_info[batch_i,:,:,:] = F.conv2d(i[batch_i,:,:,:].unsqueeze(0), textual_kernels[:,batch_i,:], bias=None)
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

        # Concatenate MM vector with Visual Vector with Text embeddings
        # x = torch.cat((x, xi), dim=1) # 2048 + 2048
        x = torch.cat((x, it), dim=1) # 2048 + 50
        x = torch.cat((x, tt), dim=1) # 2048 + 50 + 50

        x = self.fc1_mm(x)
        x = self.fc2_mm(x)
        x = self.fc3_mm(x)

        return x



class ImageEmbeddings(nn.Module):

    def __init__(self,c):
        super(ImageEmbeddings, self).__init__()

        self.fc1 = BasicFC(2048, 200)
        self.fc2 = nn.Linear(200, c['num_classes'])

    def forward(self, i, tt, it):
        x = self.fc1(i)
        # x = self.fc2(x)
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