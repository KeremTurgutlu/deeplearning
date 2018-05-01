import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import  kaiming_normal

def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)


class CrossDenseNN(nn.Module):
    # https://arxiv.org/pdf/1708.05123.pdf
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, cross_depth=6,
                 y_range=None, use_bn=False, is_reg=True, is_multi=False):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_szs])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.cross_depth = cross_depth
        # dnn layers
        szs = [n_emb + n_cont] + szs
        self.szs = szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in range(len(szs) - 1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: kaiming_normal(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        kaiming_normal(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn, self.y_range = use_bn, y_range
        self.is_reg = is_reg
        self.is_multi = is_multi

        # cross layers
        self.lins2 = nn.ModuleList([nn.Linear(self.n_emb + self.n_cont, 1).cuda()
                                    for i in range(self.cross_depth)])
        self.l_out = nn.Linear((self.n_emb + self.n_cont) + self.szs[-1], 2).cuda()

    def forward(self, x_cat, x_cont):

        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont)
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2

        # DNN
        x_dnn = x
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x_dnn = F.relu(l(x_dnn))
            if self.use_bn: x = b(x)
            x_dnn = d(x_dnn)

            # CROSS NN
        xl = x
        x0 = x
        for l in self.lins2:
            xl = l(torch.bmm(x0.unsqueeze(2), xl.unsqueeze(1))).squeeze() + xl  # bs x p

        return F.log_softmax(self.l_out(torch.cat([x_dnn, xl], 1)))