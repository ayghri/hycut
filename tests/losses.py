from torch.special import entr as entropy
from torch.nn.functional import cross_entropy
from simcspr.layers import PerFeatLinear
import torch

ins_features = [3, 4, 5]  # F
F = len(ins_features)
out_features = 2  # K
B = 10
model = PerFeatLinear(ins_features, out_features)
model2 = PerFeatLinear(ins_features, out_features)
feats_per_space = [torch.randn(B, d) for d in ins_features]
outer_logits_per_space = model(feats_per_space).transpose(1, 2)
inner_logits_per_space = model2(feats_per_space).transpose(1, 2)  # shape (B, K, F)
print(outer_logits_per_space.shape)
labels_per_space = torch.softmax(outer_logits_per_space, dim=1)
labels = labels_per_space.mean(dim=-1)
# print(outer_logits_per_space[:1], labels_per_space[:1])
# print(labels_per_space.sum(dim=-1))
print(labels.shape, inner_logits_per_space.shape)
a = 0
for i in range(len(ins_features)):
    a += cross_entropy(inner_logits_per_space[:, :, i], labels)
b = (
    cross_entropy(
        inner_logits_per_space,
        labels.unsqueeze(-1).repeat_interleave(len(ins_features), dim=-1),
    )
    * F
)
print(a)
print(b)
