import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import customDataset
import trainingFunctions as t
import mymodel

from pylab import zeros, arange, subplots, plt, savefig

training_id = 'HateSPic_inceptionv3_6fc_bs32_decay200_all'
dataset = '../../../datasets/HateSPic/HateSPic/' # Path to dataset
split_train = 'lstm_embeddings_train_hate.txt'
split_val =  'lstm_embeddings_val_hate.txt'
ImgSize = 299
gpus = [0]
gpu = 0
workers = 12 # Num of data loading workers
epochs = 301
start_epoch = 0 # Useful on restarts
batch_size = 32 #256 # Batch size
lr = 0.001 #0.01 Initial learning rate # Default 0.1, but people report better performance with 0.01 and 0.001
decay_every = 200 # Decay lr by a factor of 10 every decay_every epochs
momentum = 0.9
weight_decay = 1e-4
print_freq = 1
resume = None #dataset + '/models/resnet101_BCE/resnet101_BCE_epoch_12.pth.tar' # Path to checkpoint top resume training
# evaluate = False # Evaluate model on validation set at start
plot = True
best_prec1 = 0

weights = [0.45918, 1.0] #[0.32, 1.0] #0.3376
class_weights = torch.FloatTensor(weights).cuda()


model = mymodel.MyModel()

model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpu)

# Freeze layers
# for param in model.parameters():
#     param.requires_grad = False # This would froze all net
# Parameters of newly constructed modules have requires_grad=True by default

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(gpu)
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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['train_acc'] = zeros(epochs)
plot_data['train_acc_hate'] = zeros(epochs)
plot_data['train_acc_notHate'] = zeros(epochs)
plot_data['train_acc_avg'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['val_acc'] = zeros(epochs)
plot_data['val_acc_hate'] = zeros(epochs)
plot_data['val_acc_notHate'] = zeros(epochs)
plot_data['val_acc_avg'] = zeros(epochs)

plot_data['epoch'] = 0

# if evaluate:
#     t.validate(val_loader, model, criterion, print_freq, plot_data)

it_axes = arange(epochs)

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss (r), val loss (y), train acc hate (c), train acc not hate (o)')
ax2.set_ylabel('train acc avg (b), val acc avg (g), val acc hate (k), val acc not hate (m)')
ax2.set_autoscaley_on(False)
ax1.set_ylim([0.5, 0.8])
ax2.set_ylim([-1, 101])


for epoch in range(start_epoch, epochs):
    plot_data['epoch'] = epoch
    lr = t.adjust_learning_rate(optimizer, epoch, lr, decay_every)

    # train for one epoch
    plot_data = t.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu)

    # evaluate on validation set
    plot_data = t.validate(val_loader, model, criterion, print_freq, plot_data, gpu)

    # remember best prec@1 and save checkpoint
    is_best = plot_data['val_acc_avg'][epoch] > best_prec1
    if is_best:
        print("New best model. Prec1 = " + str(plot_data['val_acc_avg'][epoch]))
        best_prec1 = max(plot_data['val_acc_avg'][epoch], best_prec1)
        t.save_checkpoint(dataset, model, is_best, filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch))

    if plot:
        ax1.plot(it_axes[0:epoch], plot_data['train_loss'][0:epoch], 'r')
        ax2.plot(it_axes[0:epoch], plot_data['train_acc_avg'][0:epoch], 'b')

        ax2.plot(it_axes[0:epoch], plot_data['train_acc_hate'][0:epoch], 'c')
        ax2.plot(it_axes[0:epoch], plot_data['train_acc_notHate'][0:epoch], '#DBA901')

        ax1.plot(it_axes[0:epoch], plot_data['val_loss'][0:epoch], 'y')
        ax2.plot(it_axes[0:epoch], plot_data['val_acc_avg'][0:epoch], 'g')

        ax2.plot(it_axes[0:epoch], plot_data['val_acc_hate'][0:epoch], 'k')
        ax2.plot(it_axes[0:epoch], plot_data['val_acc_notHate'][0:epoch], 'm')

        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

        if epoch % 25 == 0:
            title = dataset +'/models/training/' + training_id + '_epoch_' + str(epoch) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')

