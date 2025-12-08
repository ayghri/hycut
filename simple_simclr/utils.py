import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from tqdm import tqdm

class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        eta=1e-3,
        eps=1e-8,
        clip_lr=False,
        exclude_bias_n_norm=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta,
            eps=eps,
            clip_lr=clip_lr,
            exclude_bias_n_norm=exclude_bias_n_norm,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            eta = group["eta"]
            eps = group["eps"]
            clip_lr = group["clip_lr"]
            exclude_bias_n_norm = group["exclude_bias_n_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if exclude_bias_n_norm and (p.ndim == 1 or p.shape[0] == 1):
                    # Bias and Norms
                    weight_decay = 0

                if weight_decay != 0:
                    p_norm = p.norm(2)
                    g_norm = d_p.norm(2)
                    
                    if p_norm == 0 or g_norm == 0:
                        local_lr = 1.0
                    else:
                        local_lr = eta * p_norm / (g_norm + weight_decay * p_norm + eps)
                    
                    if clip_lr:
                        local_lr = min(local_lr, 1.0)
                        
                    actual_lr = local_lr * group["lr"]
                    
                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p = d_p.mul(local_lr)
                else:
                    actual_lr = group["lr"]

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-actual_lr)

        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

@torch.no_grad()
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def compute_knn(model, train_loader, test_loader, device, k=20, t=0.1):
    model.eval()
    classes = len(train_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    
    # Extract features from training data
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for data in tqdm(train_loader, desc='Extracting train features'):
            # Handle both SimCLR dataset (returns img1, img2, target) and standard dataset (img, target)
            if len(data) == 3:
                img, _, target = data
            else:
                img, target = data
                
            img = img.to(device)
            target = target.to(device)
            feature = model(img)['feats']
            feature = F.normalize(feature, dim=1)
            train_features.append(feature)
            train_labels.append(target)
            
    train_features = torch.cat(train_features, dim=0).t().contiguous()
    train_labels = torch.cat(train_labels, dim=0).view(1, -1)
    
    # Loop test data to predict the label by weighted knn search
    with torch.no_grad():
        for data in tqdm(test_loader, desc='kNN evaluation'):
            if len(data) == 3:
                img, _, target = data
            else:
                img, target = data
                
            img = img.to(device)
            target = target.to(device)
            feature = model(img)['feats']
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, train_features, train_labels, classes, k, t)
            
            total_num += img.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            total_top5 += (pred_labels[:, :5] == target.unsqueeze(1)).sum().float().item()

    return total_top1 / total_num * 100, total_top5 / total_num * 100
