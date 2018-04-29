import sys
import torch

sys.path.append("../../fastai/")

from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
#torch.cuda.set_device(0)








class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2):
        self.ds,self.y2 = ds,y2
        self.sz = ds.sz
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))

trn_ds2 = ConcatLblDataset(md.trn_ds, trn_mcs)
val_ds2 = ConcatLblDataset(md.val_ds, val_mcs)
md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2

#######################
#####    MODEL   ######
#######################


def conv_bn_relu(kernel_size=3,stride=1, pad=1,in_c=256, out_c=256, use_bn=True):
    """Conv batchnorm relu block"""
    block = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, pad),
        nn.ReLU())
    if use_bn: block.add_module('3', nn.BatchNorm2d(in_c))
    
    # N(0, 0.01) bias = 0 initialization
    block[0].weight.data.normal_(0, 0.01)
    block[0].bias.data.zero_()
    
    return block


def flatten_conv(x, A):
    """
    IMPORTANT: Receptive fields should match target
    A: number of anchors
    
    Flatten output as:
    grid row 0 col 0 anchor 0
    grid row 0 col 0 anchor 1
    ...
    grid row 0 col 1 anchor 0
    grid row 0 col 1 anchor 1
    ...
    grid row 0 col 2 anchor 0
    grid row 0 col 2 anchor 1
    ...
    grid row n col n anchor A
    grid row n col n anchor A-1
    """
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//A)


class Subnet1(nn.Module):
    """For classification: outputs K*A"""
    def __init__(self, K, A, in_c, use_bn=False, depth=4, pi=0.01):
        super().__init__()
        
        # Number of anchors
        self.A = A
        
        # 4 block of convolutions
        self.conv = nn.Sequential(*children(conv_bn_relu(use_bn=False))*depth)
        
        # Final convolutio for prediction
        self.out_conv = nn.Conv2d(in_c, K*A, kernel_size=3, stride=1, padding=1)    
        
        # N(0, 0.01) bias = -np.log((1-pi)/pi) initialization 
        self.out_conv.weight.data.normal_(0, 0.01)
        self.out_conv.bias.data = self.out_conv.bias.data.zero_() - np.log((1-pi)/pi)
        
    def forward(self, x):
        return flatten_conv(self.out_conv(self.conv(x)), self.A)


class Subnet2(nn.Module):
    """For classification: outputs 4*A"""
    def __init__(self, A, in_c, use_bn=False, depth=4):
        super().__init__()
        
        # Number of anchors
        self.A = A
        
        # 4 block of convolutions
        self.conv = nn.Sequential(*children(conv_bn_relu(use_bn=False))*depth)
        
        # Final convolutio for prediction
        self.out_conv = nn.Conv2d(in_c, 4*A, kernel_size=3, stride=1, padding=1)    
        
        # N(0, 0.01) bias = 0
        self.out_conv.weight.data.normal_(0, 0.01)
        self.out_conv.bias.data = self.out_conv.bias.data.zero_()

    def forward(self, x):
        return flatten_conv(self.out_conv(self.conv(x)), self.A)
    
    
# load defined model
def get_encoder(f, cut):
    base_model = (cut_model(f(True), cut))
    return nn.Sequential(*base_model)


class SaveFeatures():
    """ Extract pretrained activations"""
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


class FPN(nn.Module):
def __init__(self, encoder, out_c=256):
    super().__init__()
    self.encoder = encoder        
    self.sfs = [SaveFeatures(self.encoder[i]) for i in range(len(children(self.encoder)))]
    self.out_c = out_c

def forward(self, x):
    #pdb.set_trace()
    # encode image with ResNet backbone
    x = self.encoder(x)

    # get c1, c2, c3, c4, c5 activations
    c1 = self.sfs[2].features  #64 (sz/2)
    c2 = self.sfs[4].features  #256 (sz/4)
    c3 = self.sfs[5].features  #512 (sz/8)
    c4 = self.sfs[6].features  #1024 (sz/16)
    c5 = self.sfs[7].features  #2048 (sz/32) : sz should be divisible by 32       
    C_sz = c5.size()[1] # 2048

    # construct convs
    if not hasattr(self, 'P6_conv1'):
        # get channel size of each intermediate activation
        self.sfs_c_sz = [stage_act.features.size()[1] for stage_act in self.sfs]

        self.P6_conv1 = nn.Conv2d(C_sz, self.out_c, kernel_size=3, stride=2, padding=1).cuda()
        self.P6_conv2 = nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=1, padding=1).cuda()
        self.P7_conv = nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=2, padding=1).cuda()

        self.P5_conv1 = nn.Conv2d(C_sz, self.out_c, kernel_size=1, stride=1, padding=0).cuda()
        self.P5_conv2 = nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=1, padding=1).cuda()

        self.P4_conv1 = nn.Conv2d(C_sz//2, self.out_c, kernel_size=1, stride=1, padding=0).cuda()
        self.P4_conv2 = nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=1, padding=1).cuda()

        self.P3_conv1 = nn.Conv2d(C_sz//4, self.out_c, kernel_size=1, stride=1, padding=0).cuda()
        self.P3_conv2 = nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=1, padding=1).cuda()

        #self.P2_conv1 = nn.Conv2d(C_sz//8, self.out_c, kernel_size=1, stride=1, padding=0).cuda()
        #self.P2_conv2 = nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=1, padding=1).cuda()    

    # get P2, P3, P4, P5, P6
    p6 = self.P6_conv1(c5)
    p6_out = self.P6_conv2(p6)

    p7_out = self.P7_conv(F.relu(p6_out))

    p5 = self.P5_conv1(c5) + F.upsample(p6, scale_factor=2, mode='nearest')
    p5_out = self.P5_conv2(p5)

    p4 = self.P4_conv1(c4) + F.upsample(p5, scale_factor=2, mode='nearest')
    p4_out = self.P4_conv2(p4)

    #p3 = self.P3_conv1(c3) + F.upsample(p4, scale_factor=2, mode='nearest')
    #p3_out = self.P3_conv2(p3)

    #p2 = self.P2_conv1(c2) + F.upsample(p3, scale_factor=2, mode='nearest')
    #p2_out = self.P2_conv2(p2)

    #return [p2_out, p3_out, p4_out, p5_out, p6_out, p7_out]
    #return [p3_out, p4_out, p5_out, p6_out, p7_out]
    return [p4_out, p5_out, p6_out, p7_out]




class RetinaNet(nn.Module):
    def __init__(self, fpn, subnet1, subnet2):
        super().__init__()
        self.fpn = fpn # initialize FPN
        self.subnet1 = subnet1 # initialize classifier
        self.subnet2 = subnet2 # initialize regressor
    
    def forward(self, x):
        #p3_out, p4_out, p5_out, p6_out, p7_out = self.fpn(x)
        p4_out, p5_out, p6_out, p7_out = self.fpn(x)
        
        #cls_out5 = self.subnet1(p3_out) # 32x32 
        #cls_out4 = self.subnet1(p4_out) # 16x16 
        cls_out3 = self.subnet1(p5_out) # 8x8 
        cls_out2 = self.subnet1(p6_out) # 4x4 
        cls_out1 = self.subnet1(p7_out) # 2x2 
        
        #reg_out5 = self.subnet2(p3_out) # 32x32 
        #reg_out4 = self.subnet2(p4_out) # 16x16 
        reg_out3 = self.subnet2(p5_out) # 8x8 
        reg_out2 = self.subnet2(p6_out) # 4x4 
        reg_out1 = self.subnet2(p7_out) # 2x2 
        
        
        # Concat outputs from different levels of the pyramid
        # Here, each output is coming from flatten_conv in this order:
        # grid row 0 col 0 anchor 0 level: 4x4
        # grid row 0 col 1 anchor 0 level: 4x4
        # ...
        # grid row 0 col 0 anchor 1 level: 4x4
        # grid row 0 col 1 anchor 1 level: 4x4
        #
        # grid row 0 col 0 anchor 0 level: 8x8
        # grid row 0 col 1 anchor 0 level: 8x8
        
        # grid row 0 col 0 anchor 0 level: 16x16
        # grid row 0 col 0 anchor 0 level: 16x16
        
        
        
        #return [torch.cat([cls_out1, cls_out2, cls_out3, cls_out4, cls_out5], 1),
        #       torch.cat([reg_out1, reg_out2, reg_out3, reg_out4, reg_out5], 1)]
#         return [torch.cat([cls_out1, cls_out2, cls_out3, cls_out4], 1),
#                torch.cat([reg_out1, reg_out2, reg_out3, reg_out4], 1)]
        return [torch.cat([cls_out1, cls_out2, cls_out3], 1),
               torch.cat([reg_out1, reg_out2, reg_out3], 1)]

# wrap for fastai 
class RetinaNetModel():
    def __init__(self, model, cut_lr, name='retinanet'):
        self.model,self.name, self.cut_lr = model, name, cut_lr

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.fpn.encoder), [self.cut_lr]))
        return [lgs[0]] + [children(self.model.fpn)[1:]] + [children(self.model.subnet1)] + [children(self.model.subnet2)]



#######################
#####   ANCHORS   #####
#######################


def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

#anc_grids = [2, 4, 8, 16, 32] 
anc_grids = [2, 4, 8] 

#anc_zooms = [1, 2**(1/3), 2**(2/3)]
anc_zooms = [1]

#anc_ratios = [(1.,2.), (1.,1), (2.,1.)]
anc_ratios = [(1., 1.)]

anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios] # n_anc_zooms * n_anc_ratios

k = len(anchor_scales)
anc_offsets = [1/(o*2) for o in anc_grids]

anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])

anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])

anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
               for ag in anc_grids])
grid_sizes = V(np.concatenate([np.array([ 1/ag       for i in range(ag*ag) for o,p in anchor_scales])
               for ag in anc_grids]), requires_grad=False).unsqueeze(1)
anchors = V(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()
anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])


#######################
#####   LOSS      #####
#######################

    
def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, preds, targets):
        t = one_hot_embedding(targets, self.num_classes+1)
        t = V(t[:,:-1].contiguous()) #bg class is predicted when none of the others go out.
        x = preds[:,:]
        w = self.get_weight(x,t)# for the last part
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes
    
    def get_weight(self,x,t):
        return None

class FocalLoss(BCE_Loss):
    def get_weight(self,x,t):
        alpha,gamma = 0.25,2.
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = alpha*t + (1-alpha)*(1-t)
        return w * (1-pt).pow(gamma)

loss_f = FocalLoss(len(id2cat))


def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def box_sz(b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union

#Removes the zero padding in the target bbox/class
size = 256
def get_y(bbox,clas):
    bbox = bbox.view(-1,4)/size
    bb_keep = ((bbox[:,2] - bbox[:,0])>0.).nonzero()[:,0]
    return bbox[bb_keep], clas[bb_keep]
    
def actn_to_bb(actn, anchors):
    actn_bbs = torch.tanh(actn)
    actn_ctrs = (actn_bbs[:,:2] * grid_sizes/2) + anchors[:,:2]
    actn_hw = (1 + actn_bbs[:,2:]/2) * anchors[:,2:]
    return hw2corners(actn_ctrs,actn_hw)

def map_to_ground_truth(overlaps, print_it=False):
    prior_overlap, prior_idx = overlaps.max(1)
    if print_it: print(prior_overlap)
#     pdb.set_trace()
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i,o in enumerate(prior_idx): gt_idx[o] = i
    return gt_overlap,gt_idx

def ssd_1_loss(b_c,b_bb,bbox,clas,print_it=False, use_ab=True):
    bbox,clas = get_y(bbox,clas)
    a_ic = actn_to_bb(b_bb, anchors)
    overlaps = jaccard(bbox.data, (anchor_cnr if use_ab else a_ic).data)
    gt_overlap,gt_idx = map_to_ground_truth(overlaps,print_it)
    gt_clas = clas[gt_idx]
    pos = gt_overlap > 0.5
    pos_idx = torch.nonzero(pos)[:,0]
    gt_clas[1-pos] = len(id2cat)
    gt_bbox = bbox[gt_idx]
    loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
    clas_loss  = loss_f(b_c, gt_clas) / len(pos_idx) #Normalized by the number of anchors matched to a GT object
    return loc_loss, clas_loss

def ssd_loss(pred,targ,print_it=False):
    lcs,lls = 0.,0.
    for b_c,b_bb,bbox,clas in zip(*pred,*targ):
        loc_loss,clas_loss = ssd_1_loss(b_c,b_bb,bbox,clas,print_it)
        lls += loc_loss
        lcs += clas_loss
    if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
    return lls+lcs

def ssd_loss2(pred,targ):
    lcs,lls = 0.,0.
    for b_c,b_bb,bbox,clas in zip(*pred,*targ):
        loc_loss,clas_loss = ssd_1_loss(b_c,b_bb,bbox,clas,use_ab=False)
        lls += loc_loss
        lcs += clas_loss
    return lls+lcs



#######################
#####   TRAINING  #####
#######################


# INIT MODEL DATA
PATH = Path('../../data/pascal')
trn_j = json.load((PATH/'pascal_train2007.json').open())
IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']
FILE_NAME,ID,IMG_ID,CAT_ID,BBOX = 'file_name','id','image_id','category_id','bbox'
cats = {o[ID]:o['name'] for o in trn_j[CATEGORIES]}
trn_fns = {o[ID]:o[FILE_NAME] for o in trn_j[IMAGES]}
trn_ids = [o[ID] for o in trn_j[IMAGES]]
JPEGS = 'VOCdevkit/VOC2007/JPEGImages'
IMG_PATH = PATH/JPEGS

def hw_bb(bb): return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])
trn_anno = collections.defaultdict(lambda:[])

for o in trn_j[ANNOTATIONS]:
    if not o['ignore']:
        bb = o[BBOX]
        bb = hw_bb(bb)
        trn_anno[o[IMG_ID]].append((bb,o[CAT_ID]))
    else:pass

im_dict = trn_j[IMAGES][i]
im_dict[FILE_NAME],im_dict[ID]
# open image
im = open_image(IMG_PATH/im_dict[FILE_NAME])
# get annotations
im_anno = trn_anno[im_dict[ID]]

# create multiclass classification csv
MC_CSV = PATH/'csv/mc.csv'

mc = [set([cats[p[1]] for p in trn_anno[o]]) for o in trn_ids]
mcs = [' '.join(str(p) for p in o) for o in mc]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'clas': mcs}, columns=['fn','clas'])

df.to_csv(MC_CSV, index=False)
        
# create class and bbox csv s
CLAS_CSV = PATH/'csv/clas.csv'
MBB_CSV = PATH/'csv/mbb.csv'

mc = [[cats[p[1]] for p in trn_anno[o]] for o in trn_ids]
id2cat = list(cats.values())
cat2id = {v:k for k,v in enumerate(id2cat)}
mcs = np.array([np.array([cat2id[p] for p in o]) for o in mc]); mcs

val_idxs = get_cv_idxs(len(trn_fns))
((val_mcs,trn_mcs),) = split_by_idx(val_idxs, mcs)

mbb = [np.concatenate([p[0] for p in trn_anno[o]]) for o in trn_ids]
mbbs = [' '.join(str(p) for p in o) for o in mbb]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': mbbs}, columns=['fn','bbox'])
df.to_csv(MBB_CSV, index=False)




bs = 64
aug_tfms = [RandomRotate(10, tfm_y = TfmType.COORD),
           RandomLighting(0.05,0.05, tfm_y = TfmType.COORD),
           RandomFlip(tfm_y = TfmType.COORD)]
tfms = tfms_from_model(f_model, sz, aug_tfms=aug_tfms, crop_type=CropType.NO, tfm_y = TfmType.COORD)
md = ImageClassifierData.from_csv(PATH, JPEGS, MBB_CSV, tfms=tfms, continuous=True, num_workers=4, bs=bs)

class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2):
        self.ds,self.y2 = ds,y2
        self.sz = ds.sz
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))

trn_ds2 = ConcatLblDataset(md.trn_ds, trn_mcs)
val_ds2 = ConcatLblDataset(md.val_ds, val_mcs)
md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2



# INIT MODEL
f_model=resnet34
sz=256

cut, cut_lr = model_meta[f_model]
encoder = get_encoder(f_model, cut).cuda()

# input channel for subnets, C = 256 in the paper
A, K, in_c = 1, 20, 256
fpn, subnet1, subnet2 = FPN(encoder), Subnet1(K, A, in_c), Subnet2(A, in_c)
inp = V(torch.ones(1,3,256,256))
out = fpn(inp.cuda())

retina = RetinaNet(fpn, subnet1, subnet2).cuda()
model = RetinaNetModel(retina, 8)

# init learner and define optimizer 
learn = ConvLearner(md, model)
learn.opt_fn=partial(optim.SGD,momentum=0.9)
learn.crit = ssd_loss

# freeze resnet
learn.freeze_to(1)



