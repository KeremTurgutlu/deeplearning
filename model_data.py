from torch_imports import *

### Create Columnar Dataset

# X: input numpy array
# y: target numpy array

class SimpleColumnDataset(Dataset):
    # Expecting a numpy array
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.y[index, None])

    def __len__(self):
        return len(self.X)


### Create Columnar Model Data

# X: input numpy array
# y: target numpy array
# bs: batch size
# cv: cross-validation index pairs  [(trn_idx1, val_idx1), (trn_idx2, val_idx2), .. , ..]
# test: test numpy array
# shuffe: whether to shuffle data when creating dataloaders

# This object has two modes: cv_dls and trn_dl which will be trigger
# training differently

class SimpleColumnModelData:
    def __init__(self, X, y, bs, cv=None, test=None, shuffle=True):
        if cv:
            cv_dls = []
            for cv_idxs in cv:
                trn_idx = cv_idxs[0]
                val_idx = cv_idxs[1]
                trn_val_dl = (
                DataLoader(SimpleColumnDataset(X[trn_idx], y[trn_idx]), batch_size=bs, shuffle=shuffle, num_workers=1),
                DataLoader(SimpleColumnDataset(X[val_idx], y[val_idx]), batch_size=bs, shuffle=shuffle, num_workers=1))
                cv_dls.append(trn_val_dl)

            self.cv_dls = cv_dls

        else:
            self.trn_dl = DataLoader(SimpleColumnDataset(X, y), batch_size=bs, shuffle=shuffle, num_workers=1)
        if test is not None: self.test = torch.from_numpy(test)

    @classmethod
    def from_dataframe(cls, dataframe, bs, cv=None, test=None, target=None, shuffle=True):
        X, y = np.array(dataframe.drop(target, 1)), np.array(dataframe[target])
        if test is not None: test = np.array(test)
        return cls(X, y, bs, cv, test, shuffle=True)