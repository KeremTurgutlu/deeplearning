# misc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import tqdm

# pytorch
from torch.utils.data import DataLoader as torch_dl
from torch.utils.data import Dataset
from torch import nn
from torch import optim
from torch.nn.init import *
from torch.nn import functional as F

# read data

train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')


#########################################################################################################
# FUNCTIONS
#########################################################################################################

def preprocess(data):
    data.fillna('missing', inplace=True)
    data.category_name = data.category_name.astype('category').cat.codes
    data.brand_name = data.brand_name.astype('category').cat.codes
    data.item_condition_id = data.item_condition_id
    return data


def EmbeddingDataPreprocess(data, cats, inplace=True):
    ### Each categorical column should have indices as values
    ### Which will be looked up at embedding matrix and used in modeling
    ### Make changes inplace
    if inplace:
        for c in cats:
            data[c].replace({val: i for i, val in enumerate(data[c].unique())}, inplace=True)
        return data
    else:
        data_copy = data.copy()
        for c in cats:
            data_copy[c].replace({val: i for i, val in enumerate(data_copy[c].unique())}, inplace=True)
        return data_copy


def get_embs_dims(data, cats):
    cat_sz = [len(data[c].unique()) for c in cats]
    return [(c, min(50, (c + 1) // 2)) for c in cat_sz]


def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)


#########################################################################################################
# CLASSES
#########################################################################################################

class EmbeddingDataset(Dataset):
    ### This dataset will prepare inputs cats, conts and output y
    ### To be feed into our mixed input embedding fully connected NN model
    ### Stacks numpy arrays to create nxm matrices where n = rows, m = columns
    ### Gives y 0 if not specified
    def __init__(self, cats, conts, y):
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n, 1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n, 1))
        self.y = np.zeros((n, 1)) if y is None else y[:, None].astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont, y=None):
        cat_cols = [c.values for n, c in df_cat.items()]
        cont_cols = [c.values for n, c in df_cont.items()]
        return cls(cat_cols, cont_cols, y)

    @classmethod
    def from_data_frame(cls, df, cat_flds, y=None):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y)



        ### We will keep this for fastai compatibility


class ModelData():
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path, self.trn_dl, self.val_dl, self.test_dl = path, trn_dl, val_dl, test_dl


class EmbeddingModelData(ModelData):
    ### This class provides training and validation dataloaders
    ### Which we will use in our model

    def __init__(self, path, trn_ds, val_ds, bs, test_ds=None):
        test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
        super().__init__(path, torch_dl(trn_ds, batch_size=bs, shuffle=True, num_workers=1)
                         , torch_dl(val_ds, batch_size=bs, shuffle=True, num_workers=1), test_ds)

    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, test_df=None):
        test_ds = EmbeddingDataset.from_data_frame(test_df, cat_flds) if test_df is not None else None
        return cls(path, EmbeddingDataset.from_data_frame(trn_df, cat_flds, trn_y),
                   EmbeddingDataset.from_data_frame(val_df, cat_flds, val_y), bs, test_ds=test_ds)

    @classmethod
    def from_data_frame(cls, path, val_idxs, trn_idxs, df, y, cat_flds, bs, test_df=None):
        val_df, val_y = df.iloc[val_idxs], y[val_idxs]
        trn_df, trn_y = df.iloc[trn_idxs], y[trn_idxs]
        return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs, test_df)


class EmbeddingModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range=None, use_bn=False, classify=None):
        super().__init__()  ## inherit from nn.Module parent class
        self.embs = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_szs])  ## construct embeddings
        for emb in self.embs: emb_init(emb)  ## initialize embedding weights
        n_emb = sum(e.embedding_dim for e in self.embs)  ## get embedding dimension needed for 1st layer
        szs = [n_emb + n_cont] + szs  ## add input layer to szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in
            range(len(szs) - 1)])  ## create linear layers input, l1 -> l1, l2 ...
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])  ## batchnormalization for hidden layers activations
        for o in self.lins: kaiming_normal(o.weight.data)  ## init weights with kaiming normalization
        self.outp = nn.Linear(szs[-1], out_sz)  ## create linear from last hidden layer to output
        kaiming_normal(self.outp.weight.data)  ## do kaiming initialization

        self.emb_drop = nn.Dropout(emb_drop)  ## embedding dropout, will zero out weights of embeddings
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])  ## fc layer dropout
        self.bn = nn.BatchNorm1d(n_cont)  # bacthnorm for continous data
        self.use_bn, self.y_range = use_bn, y_range
        self.classify = classify

    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]  # takes necessary emb vectors
        x = torch.cat(x, 1)  ## concatenate along axis = 1 (columns - side by side) # this is our input from cats
        x = self.emb_drop(x)  ## apply dropout to elements of embedding tensor
        x2 = self.bn(x_cont)  ## apply batchnorm to continous variables
        x = torch.cat([x, x2], 1)  ## concatenate cats and conts for final input
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))  ## dotprod + non-linearity
            if self.use_bn: x = b(x)  ## apply batchnorm activations
            x = d(x)  ## apply dropout to activations
        x = self.outp(x)  # we defined this externally just not to apply dropout to output
        if self.classify:
            x = F.sigmoid(x)  # for classification
        elif y_range:
            x = F.sigmoid(x)  ## scales the output between 0,1
            x = x * (self.y_range[1] - self.y_range[0])  ## scale output
            x = x + self.y_range[0]  ## shift output
        return x


#########################################################################################################
# RUN MODEL
#########################################################################################################



train.columns = ['id', 'name', 'item_condition_id', 'category_name', 'brand_name',
                 'price', 'shipping', 'item_description']

test['price'] = 0
test.columns = ['id', 'name', 'item_condition_id', 'category_name', 'brand_name',
                'shipping', 'item_description', 'price']

train_test = pd.concat([train, test], 0)
train_test.drop(['id', 'name', 'item_description'], axis=1, inplace=True)
train_test = preprocess(train_test)
train_test = train_test.reset_index(drop=True)
cats = ['item_condition_id', 'category_name', 'brand_name']
train_test = EmbeddingDataPreprocess(train_test, cats, inplace=True)
train_df = train_test.iloc[range(len(train))]
test_df = train_test.iloc[range(len(train), len(train_test))]

del train
test_id = test['id']
del test
gc.collect()

train_input, train_y = train_df.drop('price', 1), np.log(train_df.price + 1)
test_input, test_y = test_df.drop('price', 1), np.log(test_df.price + 1)
y_range = (train_y.min(), train_y.max())
emb_szs = get_embs_dims(train_test, cats)

model_data = EmbeddingModelData.from_data_frames('./tmp', train_input, test_input, train_y, test_y, cats, bs=32)
emb_model = EmbeddingModel(emb_szs, 1, 0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range, classify=None)


def embedding_train(model, model_data, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for data in iter(model_data.trn_dl):
            # get inputs
            x_cats, x_conts, y = data

            # wrap with variable
            x_cats, x_conts, y = Variable(x_cats), Variable(x_conts), Variable(y)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x_cats, x_conts)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()


# First training
opt = optim.SGD(emb_model.parameters(), lr=1e-4, weight_decay=1e-4)
crit = F.mse_loss
epochs = 1
embedding_train(emb_model, model_data, opt, crit, 1)

# Second training
opt = optim.SGD(emb_model.parameters(), lr=5e-4, weight_decay=1e-4)
crit = F.mse_loss
epochs = 1
embedding_train(emb_model, model_data, opt, crit, 1)

# Make predictions
preds = emb_model(Variable(LongTensor(model_data.val_dl.dataset.cats)),
                  Variable(FloatTensor(model_data.val_dl.dataset.conts)))

pd.DataFrame({'test_id': test_id, 'price': (np.exp(preds.flatten()) - 1)}).to_csv('predictions.csv', index=False)