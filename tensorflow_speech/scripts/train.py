from imports import *

def evaluate(net, valid_dl, criterion):
    val_true = []
    val_pred = []
    for i, data in enumerate(valid_dl, 0):
        #get the inputs
        inputs, labels = data

        #wrap them in Variable
        inputs, labels = V(inputs.float().cuda()), V(labels.cuda())
        outputs = net(inputs)

        #get cross entropy loss and accuracy
        val_true.append(labels.data)
        val_pred.append(outputs.data)

    #define validation variables
    val_pred_concat = V(torch.cat(val_pred))
    val_true_concat = V(torch.cat(val_true))
    #validation cross entropy loss
    val_loss = criterion(val_pred_concat, val_true_concat).data[0]
    #validation accuracy
    _, class_pred = torch.max(val_pred_concat, 1)
    val_acc = sum((class_pred == val_true_concat).data) / len(class_pred)
    #confusion matrix
    cmat = confusion_matrix(val_true_concat.data.cpu().numpy(), class_pred.data.cpu().numpy())
    return val_loss, val_acc, cmat


def train(net, train_dl, valid_dl, criterion, optimizer, n):
    training_loss = []
    alpha = 0.98
    running_loss = 0.0
    batch_num = 0

    for epoch in range(n):  # loop over the dataset multiple times
        net.train()  # back to training
        for i, data in enumerate(train_dl, 0):
            batch_num += 1
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = V(inputs.float().cuda()), V(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # compute debias running training loss
            training_loss.append(loss.data[0])
            running_loss = running_loss * alpha + loss.data[0] * (1 - alpha)
            debias_loss = running_loss / (1 - alpha ** batch_num)

        net.eval()  # for dropout
        val_loss, val_acc, cmat = evaluate(net, valid_dl, criterion)
        # print epoch evaluation
        print(f'epoch {epoch}')
        print(f'[{debias_loss} ,{val_loss}, {val_acc}]\n')

        # save model at every 1000th step
        if batch_num % 500 == 0:
            save_model(net, batch_num, epoch, optimizer)

    save_model(net, batch_num, epoch, optimizer)
    # print final cmat
    print(cmat)
    return training_loss


def save_model(net, batch_num, epoch, optimizer):
    print(f'Saving model at step {batch_num}')
    torch.save(net.state_dict(), f'../models/trad_pool2d_{batch_num}')
    torch.save({'optimizer': optimizer.state_dict(),
                'iter': batch_num,
                'epoch': epoch}, f'../models/trad_pool2d_optim_{batch_num}')