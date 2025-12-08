import torchvision.datasets as dsets
import torchvision.transforms as transforms
from pathlib import Path
import random
import numpy as np
import torch
from torchinfo import summary
import time

# datasets_path = Path("/buckets/datasets/imagenet/official/")
datasets_path = Path("/buckets/datasets/imagenet/official/archives/")
checkpoint_dir = Path("/buckets/models/dinov2/")
model_name = "facebookresearch/dinov2"
device = torch.device("cuda")

ds_val = dsets.ImageNet(root=datasets_path, split="train")
n = len(ds_val)
len(ds_val)
# checkpoint_dir = checkpoint_dir.joinpath("dinov2")
torch.hub.set_dir(checkpoint_dir)
model = torch.hub.load(model_name, "dinov2_vitg14").to(device)
model.eval()

for _ in range(10):
    # for x,y in
    i = random.randint(0, n)
    x, y = ds_val[i]
    # ds_val:
    # print(x.size,y)
    print(f"size(x[{i}]) {x.size}, y:{y}")
    # if i > 100:
    # break
    # i+=1
# x
# y
print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
)
print(model)

summary(model, (1, 3, 224, 224))
time.sleep(5)
