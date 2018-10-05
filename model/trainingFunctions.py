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
    acc_hate = AverageMeter()
    acc_notHate = AverageMeter()
    acc_avg = AverageMeter()


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
        target_var = torch.autograd.Variable(target).squeeze(1)

        # compute output
        output = model(image_var, image_text_var, tweet_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        prec1 = accuracy(output.data, target.long().cuda(gpu), topk=(1,))
        cur_acc_hate, cur_acc_notHate = accuracy_per_class(output.data, target.long().cuda(gpu))
        acc_hate.update(cur_acc_hate, image.size()[0])
        acc_notHate.update(cur_acc_notHate, image.size()[0])
        acc_avg.update((cur_acc_hate + cur_acc_notHate) / 2, image.size()[0])
        # print image.size()[0]
        losses.update(loss.data.item(), image.size()[0])
        acc.update(prec1[0], image.size()[0])

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
                  'Acc {acc.val.data[0]:.3f} ({acc.avg.data[0]:.3f})\t'
                  'Acc Hate {acc_hate.val:.3f} ({acc_hate.avg:.3f})\t'
                  'Acc NotHate {acc_notHate.val:.3f} ({acc_notHate.avg:.3f})\t'
                  'Acc Avg {acc_avg.val:.3f} ({acc_avg.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=acc, acc_hate=acc_hate, acc_notHate=acc_notHate, acc_avg=acc_avg))

    print('TRAIN: Acc: ' + str(acc.avg.data[0].item()) + 'Acc Avg: ' + str(acc_avg.avg) + ' Hate Acc: ' + str(acc_hate.avg) + ' - Not Hate Acc: ' + str(
        acc_notHate.avg))

    plot_data['train_loss'][plot_data['epoch']] = losses.avg
    plot_data['train_acc'][plot_data['epoch']] = acc.avg
    plot_data['train_acc_hate'][plot_data['epoch']] = acc_hate.avg
    plot_data['train_acc_notHate'][plot_data['epoch']] = acc_notHate.avg
    plot_data['train_acc_avg'][plot_data['epoch']] = acc_avg.avg


    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        acc_hate = AverageMeter()
        acc_notHate = AverageMeter()
        acc_avg = AverageMeter()

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

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            prec1 = accuracy(output.data, target.long().cuda(gpu), topk=(1,))
            cur_acc_hate, cur_acc_notHate = accuracy_per_class(output.data, target.long().cuda(gpu))
            acc_hate.update(cur_acc_hate, image.size()[0])
            acc_notHate.update(cur_acc_notHate, image.size()[0])
            acc_avg.update((cur_acc_hate + cur_acc_notHate) / 2, image.size()[0])

            losses.update(loss.data.item(), image.size()[0])
            acc.update(prec1[0], image.size()[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val.data[0]:.3f} ({acc.avg.data[0]:.3f})'
                      'Acc Hate {acc_hate.val:.3f} ({acc_hate.avg:.3f})\t'
                      'Acc NotHate {acc_notHate.val:.3f} ({acc_notHate.avg:.3f})\t'
                      'Acc Avg {acc_avg.val:.3f} ({acc_avg.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       acc=acc, acc_hate=acc_hate, acc_notHate=acc_notHate, acc_avg=acc_avg))

        print('VALIDATION: Acc: ' + str(acc.avg.data[0].item()) + 'Acc Avg: ' + str(acc_avg.avg) + ' Hate Acc: ' + str(acc_hate.avg) + ' - Not Hate Acc: ' + str(acc_notHate.avg ))

        plot_data['val_loss'][plot_data['epoch']] = losses.avg
        plot_data['val_acc'][plot_data['epoch']] = acc.avg
        plot_data['val_acc_hate'][plot_data['epoch']] = acc_hate.avg
        plot_data['val_acc_notHate'][plot_data['epoch']] = acc_notHate.avg
        plot_data['val_acc_avg'][plot_data['epoch']] = acc_avg.avg


    return plot_data


def save_checkpoint(dataset, model, is_best, filename='checkpoint.pth.tar'):
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

    # if epoch != 0 and epoch % decay_every == 0:
        # lr = lr * 0.1
        # print("Learning rate changed to " + str(lr))
    print("Learning rate reduced by 10")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
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


def accuracy_per_class(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    correct_hate = 0
    correct_notHate = 0
    total_hate = 0
    total_notHate = 0
    pred = pred[0]

    for i, cur_target in enumerate(target):

        if cur_target == 1:
            total_hate += 1
            if cur_target == pred[i]: correct_hate += 1
        else:
            total_notHate += 1
            if cur_target == pred[i]: correct_notHate += 1

    if total_hate == 0 : total_hate = 1
    if total_notHate == 0 : total_notHate = 1

    acc_hate = 100 * float(correct_hate) / total_hate
    acc_notHate = 100 * float(correct_notHate) / total_notHate


    return acc_hate, acc_notHate

