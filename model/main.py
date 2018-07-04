import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import customDataset
import trainingFunctions as t
import torch.nn.functional as F


from pylab import zeros, arange, subplots, plt, savefig

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

training_id = 'HateSPic_inception_v3_bs32'
dataset = '../../../datasets/HateSPic/HateSPic/' # Path to dataset
split_train = 'lstm_scores_train_hate.txt'
split_val =  'lstm_scores_val_hate.txt'
arch = 'inception_v3'
ImgSize = 299
gpus = [0]
gpu = 0
workers = 12 # Num of data loading workers
epochs = 150
start_epoch = 0 # Useful on restarts
batch_size = 32 #256 # Batch size
lr = 0.001 #0.01 Initial learning rate # Default 0.1, but people report better performance with 0.01 and 0.001
decay_every = 20 # Decay lr by a factor of 10 every decay_every epochs
momentum = 0.9
weight_decay = 1e-4
print_freq = 500
resume = None #dataset + '/models/resnet101_BCE/resnet101_BCE_epoch_12.pth.tar' # Path to checkpoint top resume training
# evaluate = False # Evaluate model on validation set at start
plot = True
best_prec1 = 0
aux_logits = False # To desactivate the other loss in Inception v3 (there is only one extra loss


class MyModel(nn.Module):

    def __init__(self):

        num_classes = 2
        lstm_hidden_state_dim = 50

        super(MyModel, self).__init__()
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)

        # Delete last fc that maps 2048 features to 1000 classes.
        # Now the output of CNN is the 2048 features
        del(self.cnn._modules['fc'])

        # Create the linear layers that will process both the img and the txt
        self.fc1 = nn.Linear(2048 + lstm_hidden_state_dim * 2, 2048 + lstm_hidden_state_dim * 2)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, num_classes)


    def forward(self, image, img_text, tweet):
        x1 = self.cnn(image)
        x2 = img_text
        x3 = tweet

        x = torch.cat((x2, x3), dim=1)
        x = torch.cat((x1, x), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


model = MyModel()

model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpu)

# Edit model
# for param in model.parameters():
#     param.requires_grad = False # This would froze all net
# Parameters of newly constructed modules have requires_grad=True by default
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda(gpu)
# criterion = nn.MultiLabelSoftMarginLoss().cuda(gpu) # This is not the loss I want
# criterion = nn.BCEWithLogitsLoss().cuda(gpu)

optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

# optionally resume from a checkpoint
if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

cudnn.benchmark = True

# Data loading code
train_dataset = customDataset.CustomDataset(
    dataset,split_train,Rescale=0,RandomCrop=ImgSize,Mirror=True)

val_dataset = customDataset.CustomDataset(
    dataset, split_val,Rescale=ImgSize,RandomCrop=0,Mirror=False)

# if distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# else:
train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    num_workers=workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['train_acc'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['val_acc'] = zeros(epochs)
plot_data['epoch'] = 0

# if evaluate:
#     t.validate(val_loader, model, criterion, print_freq, plot_data)

it_axes = arange(epochs)

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss (r), val loss (y)')
ax2.set_ylabel('train acc (b), val acc (g)')
ax2.set_autoscaley_on(False)
ax1.set_ylim([0, 0.1])
ax2.set_ylim([0, 100])


for epoch in range(start_epoch, epochs):
    plot_data['epoch'] = epoch
    lr = t.adjust_learning_rate(optimizer, epoch, lr, decay_every)

    # train for one epoch
    plot_data = t.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu)

    # evaluate on validation set
    plot_data, prec1 = t.validate(val_loader, model, criterion, print_freq, plot_data, gpu)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    print("New best model. Prec1 = " + str(prec1))
    best_prec1 = max(prec1, best_prec1)
    t.save_checkpoint(dataset, {
        'model': model,
        'epoch': epoch,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best, filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch) + '.pth.tar')

    if plot:
        ax1.plot(it_axes[0:epoch], plot_data['train_loss'][0:epoch], 'r')
        ax2.plot(it_axes[0:epoch], plot_data['train_acc'][0:epoch], 'b')

        ax1.plot(it_axes[0:epoch], plot_data['val_loss'][0:epoch], 'y')
        ax2.plot(it_axes[0:epoch], plot_data['val_acc'][0:epoch], 'g')

        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)
        title = dataset +'/models/training/' + training_id + '_epoch_' + str(epoch) + '.png'  # Save graph to disk
        savefig(title, bbox_inches='tight')

