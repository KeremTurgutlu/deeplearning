from imports import *
from datasets import *
from cnn_models import *
from train import *
import argparse

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('--subsample', type=float, default=1.0)
parser.add_argument('--epochs', type=int)
args = parser.parse_args()
subsample = args.subsample
epochs = args.epochs
print(f'Subsample:{subsample} Epochs:{epochs}')


# here is a list of the classes to be predicted
classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence', 'unknown']

train_path = '../data/train/train/'
val_path = '../data/train/val/'

train_ds = logSpecData(train_path, classes, sample_ratio=subsample)
val_ds = logSpecData(val_path, classes, sample_ratio=subsample)

print(f'{len(train_ds.img_paths)} training samples, {len(val_ds.img_paths)} validation samples')
print(f'classes {classes}')

weights, weight_dict = get_sampler_weights(train_ds)
sampler = WeightedRandomSampler(weights, len(weights))
train_dl = DataLoader(train_ds, 128, sampler=sampler, drop_last=True, num_workers=8)
valid_dl = DataLoader(val_ds, 128, num_workers=8)


#define network
#distribute module to gpu to work on parallel
net = cnn_trad_pool2_net()
net.cuda()
print(f'Training...')
# print(f'Network ready')
# if torch.cuda.is_available():
#     net.cuda()
#     n_devices = torch.cuda.device_count()
#     print(f'Running model on {n_devices} GPUs\n')
#     net = nn.DataParallel(net)
# else: print(f'Running model on cpu')
# define optimizer and criterion
criterion = F.cross_entropy
optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0001)
#optimizer = optim.Adam(net.parameters(), lr=0.01)

training_loss = train(net, train_dl, valid_dl, criterion, optimizer, epochs)

