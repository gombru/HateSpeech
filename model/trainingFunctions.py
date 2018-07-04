import shutil
import time
import torch
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, image_text, tweet, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(gpu, async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).squeeze(1)

        # compute output
        output = model(input_var, image_text, tweet)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # for multilabel, we select a random positive class to compute accuracy, and for regression the max value
        one_target = torch.zeros([int(target.size()[0]), 1])
        for c in range(0,int(target.size()[0])):
            one_target[c] = (target[c] == target[c].max()).nonzero()[0].float()[0]
        prec1 = accuracy(output.data, one_target.long().cuda(gpu), topk=(1))

        losses.update(loss.data[0], input.size(0))
        acc.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=acc))

    plot_data['train_loss'][plot_data['epoch']] = losses.avg
    plot_data['train_acc'][plot_data['epoch']] = acc.avg

    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(gpu, async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target).squeeze(1) # Needed because we need a 1-dimension vector

            # test_target = torch.autograd.Variable(torch.LongTensor(3).random_(5).long())
            # output_test = torch.autograd.Variable(torch.randn(3, 7), requires_grad=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # for multilabel, we select a random positive class to compute accuracy, and for regression the max value
            one_target = torch.zeros([int(target.size()[0]), 1])
            for c in range(0,int(target.size()[0])):
                one_target[c] = (target[c] == target[c].max()).nonzero()[0].float()[0]
            prec1 = accuracy(output.data, one_target.long().cuda(gpu), topk=(1))

            losses.update(loss.data[0], input.size(0))
            acc.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       acc=acc))

        print(' * Acc {acc.avg:.3f}'
              .format(acc=acc))

        plot_data['val_loss'][plot_data['epoch']] = losses.avg
        plot_data['val_acc'][plot_data['epoch']] = acc.avg

    return plot_data, acc.avg


def save_checkpoint(dataset, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, dataset +'/models/' + 'model_best.pth.tar')


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
        lr = lr * 0.1
        print("Learning rate changed to " + str(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
