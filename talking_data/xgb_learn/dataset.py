import xgboost as xgb
import pandas as pd

class XGBModelData:
    def __init__(self, trn_df, val_df, trn_y, val_y, test_df=None):

        self.trn_df = trn_df
        self.val_df = val_df
        self.trn_y = trn_y
        self.val_y = val_y
        self.test_df = test_df

    def get_train_eval_ds(self):
        """Data for evaluation mode"""
        dtrain = xgb.DMatrix(self.trn_df, label=self.trn_y)
        dval = xgb.DMatrix(self.val_df, label=self.val_y)
        evals = [(dtrain, 'train'), (dval, 'valid')]
        return dtrain, dval, evals

    def get_train_test_ds(self):
        """Data for final training mode"""
        full_trn_df = pd.concat([self.trn_df, self.val_df])
        full_trn_y = pd.concat([pd.Series(self.trn_y), pd.Series(self.val_y)])

        dtrain = xgb.DMatrix(full_trn_df, label=full_trn_y)
        dtest = xgb.DMatrix(self.test_df)
        return dtrain, dtest

    @classmethod
    def from_cv_idxs(cls, df, y, cv_idxs):
        """
        df (pd.DataFrame): Dataframe for independent variable
        y (pd.Series, np.array): Dependent variable
        cv_idxs (list): list of indexes/bool masks for trn, val, test sets
        """

        if len(cv_idxs) == 3:
            test_df = df[cv_idxs[-1]]
        else:
            test_df = None
        trn_idxs, val_idxs = cv_idxs[:2]
        trn_df, val_df, trn_y, val_y = df[trn_idxs], df[val_idxs], y[trn_idxs], y[val_idxs]
        return cls(trn_df, val_df, trn_y, val_y, test_df)