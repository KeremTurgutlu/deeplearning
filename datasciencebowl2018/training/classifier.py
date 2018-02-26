import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from model.loss import *
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
    def __init__(self, net, max_epochs):
        """
        Neural Net classifier for Data Science Bowl
        Inputs:
            net (nn.Module): PyTorch model that is going to forward
            max_epochs (int) : The max number of epochs to train the model
        """

        self.net = net
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.use_cuda = torch.cuda.is_available()

    # def _criterion(self, logits, labels):
    #     return BCELoss2d().forward(logits, labels) + \
    #            SoftDiceLoss().forward(logits, labels)

    def _validate_epoch(self, valid_loader, threshold):
        """
        Validate by self._criterion
        """
        losses = AverageMeter() #reset for epoch

        batch_size = valid_loader.batch_size

        for ind, (inputs, targets, index) in enumerate(valid_loader):
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            inputs, targets = V(inputs, volatile=True), V(targets, volatile=True) # volatile since inference mode

            # forward
            logits = self.net(inputs)
            probs = F.sigmoid(logits)
            preds = (probs > threshold).double()
            # compute loss for iteration
            loss = self._criterion(logits, targets)
            # update loss for iteration
            losses.update(loss.data[0], batch_size)

        return losses.avg

    def _train_epoch(self, train_loader, optimizer, threshold):
        """
        Optimize by self._criterion and self.optimizer
        """
        losses = AverageMeter() # reset for epoch
        batch_size = train_loader.batch_size

        for ind, (inputs, targets, index) in enumerate(train_loader):

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            inputs, targets = V(inputs), V(targets)

            # forward
            logits = self.net(inputs)
            probs = F.sigmoid(logits)
            preds = (probs > threshold).double()
            # compute loss for iteration
            loss = self._criterion(logits, targets)
            # update loss for iteration
            losses.update(loss.data[0], batch_size)
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses.avg

    def _run_epoch(self, train_loader, valid_loader, optimizer,
                   threshold=0.5):
        """
        Run a single epoch print training and validation loss
        """
        # train mode
        self.net.train()

        # run a train pass
        train_loss = self._train_epoch(train_loader, optimizer, threshold)

        # eval mode
        self.net.eval()

        # validate epoch
        val_loss = self._validate_epoch(valid_loader, threshold)

        self.epoch_counter += 1

        print(f"Epoch: {self.epoch_counter}")
        print(f"Training : [{round(train_loss, 4)}],\
Validation : [{round(val_loss, 4)}]")

    # main methods
    def train(self, train_loader, valid_loader, optimizer, crit, epochs, threshold=0.5):
        """
        Takes optimizer and criterion
        """
        self.optimizer = optimizer
        self._criterion = crit

        if self.use_cuda:
            self.net.cuda()

        for epoch in range(epochs):
            self._run_epoch(train_loader, valid_loader, optimizer, threshold)


    def restore_model(self, model_path, optim_path=None):
        """
        Restore a model parameters from the given path
        Inputs:
            model_path (str): The path to the model to restore
        """
        self.net.load_state_dict(torch.load(model_path))
        if optim_path is not None:
            self.optimizer.load_state_dict(optim_path)

    def save_model(self, model_path, optim_path):
        torch.save(self.net.state_dict(), model_path)
        if optim_path is not None:
            torch.save(self.optimizer.state_dict(), optim_path)

    def predict(self, test_loader):
        # eval mode
        self.net.eval()
        preds = []
        for ind, (images, index) in enumerate(test_loader):
            if self.use_cuda:
                images = images.cuda()

            images = V(images, volatile=True)

            # forward
            logits = self.net(images)
            probs = F.sigmoid(logits)
            probs = probs.data.cpu().numpy()
            preds.append(probs)
        return preds

