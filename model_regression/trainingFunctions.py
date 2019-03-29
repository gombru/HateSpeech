import shutil
import time
import torch
import torch.nn.parallel
import glob
import os

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, image_text, tweet, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(gpu, async=True)
        image_var = torch.autograd.Variable(image)
        image_text_var = torch.autograd.Variable(image_text)
        tweet_var = torch.autograd.Variable(tweet)
        target_var = torch.autograd.Variable(target).unsqueeze(1)

        # compute output
        output = model(image_var, image_text_var, tweet_var)
        loss = criterion(output, target_var)

        losses.update(loss.data.item(), image.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU: {gpu}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), gpu=str(gpu), batch_time=batch_time,
                   data_time=data_time, loss=losses))


    plot_data['train_loss'][plot_data['epoch']] = losses.avg


    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image, image_text, tweet, target) in enumerate(val_loader):

            target = target.cuda(gpu, async=True)
            image_var = torch.autograd.Variable(image)
            image_text_var = torch.autograd.Variable(image_text)
            tweet_var = torch.autograd.Variable(tweet)
            target_var = torch.autograd.Variable(target).squeeze(1)


            # compute output
            output = model(image_var, image_text_var, tweet_var)
            loss = criterion(output, target_var)

            losses.update(loss.data.item(), image.size()[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                ))

        plot_data['val_loss'][plot_data['epoch']] = losses.avg

    return plot_data


def save_checkpoint(dataset, model, is_best, filename='checkpoint.pth.tar'):
    print("Saving Checkpoint")
    prefix = 16
    if '_ValLoss_' in filename:
        prefix = 30
    for cur_filename in glob.glob(filename[:-prefix] + '*'):
        print(cur_filename)
        os.remove(cur_filename)
    torch.save(model.state_dict(), filename + '.pth.tar')
    # if is_best:
    #     shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr, decay_every):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # lr = lr * (0.1 ** (epoch // decay_every)) # This was the former code but its wrong

    if epoch != 0 and epoch % decay_every == 0:
        # lr = lr * 0.1
        # print("Learning rate changed to " + str(lr))
        print("Learning rate reduced by 10")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return lr

