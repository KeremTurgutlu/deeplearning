import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from evals.loss import *
import matplotlib.pyplot as plt

# Average meter counter
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


class NucleiClassifier:
    def __init__(self, net, optimizer, crit, metric, gpu=1):
        """
        Neural Net classifier for Data Science Bowl
        Inputs:
            net (nn.Module): PyTorch model that is going to forward
        """

        self.net = net
        self.epoch_counter = 0  # counts number of epochs
        self.use_cuda = torch.cuda.is_available()  # enable cuda if gpu available
        self.gpu = gpu  # gpu id to train on
        self.optimizer = optimizer  # optimizer to use
        self.crit = crit  # loss function to use
        self.metric = metric  # metric to use

    def _validate_epoch(self, valid_loader, threshold):
        """
        Validate by self._criterion
        """
        losses = AverageMeter()  # reset for epoch
        metrics = AverageMeter()  # reset for epoch
        batch_size = valid_loader.batch_size

        for ind, (inputs, targets, index) in enumerate(valid_loader):
            if self.use_cuda:
                inputs = inputs.cuda(self.gpu)
                targets = targets.cuda(self.gpu)
            inputs, targets = V(inputs, volatile=True), V(targets, volatile=True)  # volatile since inference mode

            # forward
            logits = self.net(inputs)

            # compute and update loss
            loss = self.crit(logits, targets)
            losses.update(loss.data[0], batch_size)

            # compute and update metric
            metric = self.metric(logits, targets, threshold)
            metrics.update(metric.data[0], batch_size)

        return losses.avg, metrics.avg

    def _train_epoch(self, train_loader, optimizer, threshold):
        """
        Optimize by self._criterion and self.optimizer
        """
        losses = AverageMeter()  # reset for epoch
        metrics = AverageMeter()  # reset for epoch
        batch_size = train_loader.batch_size

        for ind, (inputs, targets, index) in enumerate(train_loader):

            if self.use_cuda:
                inputs = inputs.cuda(self.gpu)
                targets = targets.cuda(self.gpu)
            inputs, targets = V(inputs), V(targets)

            # forward
            logits = self.net(inputs)

            # compute and update loss
            loss = self.crit(logits, targets)
            losses.update(loss.data[0], batch_size)

            # compute and update metric
            metric = self.metric(logits, targets, threshold)
            metrics.update(metric.data[0], batch_size)

            # backprop - update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses.avg, metrics.avg

    def _run_epoch(self, train_loader, valid_loader, threshold=0.5):
        """
        Run a single epoch print training and validation loss
        """
        # train mode and run a train pass
        self.net.train()
        train_loss, train_metric = self._train_epoch(train_loader, self.optimizer, threshold)

        # eval mode and validate epoch
        self.net.eval()
        val_loss, val_metric = self._validate_epoch(valid_loader, threshold)
        self.epoch_counter += 1

        print(f"Epoch: {self.epoch_counter}")
        print(f"LOSS - Training : [{round(train_loss, 4)}], Validation : [{round(val_loss, 4)}]")
        print(f"METRIC - Training : [{round(train_metric, 4)}], Validation : [{round(val_metric, 4)}]")

    # main methods
    def train(self, train_loader, valid_loader, epochs, threshold=0.5):
        """
        ...
        """
        if self.use_cuda:
            self.net.cuda(self.gpu)

        for epoch in range(epochs):
            self._run_epoch(train_loader, valid_loader, threshold)

    def restore_model(self, path, gpu):
        """
        Restore a model parameters from the given path
        Inputs:
            model_path (str): The path to the model to restore
        """
        self.net.cpu()
        state = torch.load(path)
        self.net.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch_counter = state['epoch']  # counts number of epochs

    def save_model(self, path):
        state = {
            'epoch': self.epoch_counter,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def predict(self, test_loader):
        # eval mode
        self.net.eval()
        preds = []
        for ind, (images, index) in enumerate(test_loader):
            if self.use_cuda:
                images = images.cuda(self.gpu)

            images = V(images, volatile=True)

            # forward
            logits = self.net(images)
            probs = F.sigmoid(logits)
            probs = probs.data.cpu().numpy()
            preds.append(probs)
        return preds
