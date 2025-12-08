import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

def simclr_loss_func(
    z: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views.
    
    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        temperature (float): temperature factor for the loss.

    Return:
        torch.Tensor: SimCLR loss.
    """
    device = z.device
    z = F.normalize(z, dim=-1)
    
    # Calculate similarity matrix
    # z is (2*N, D)
    # sim is (2*N, 2*N)
    sim = torch.exp(torch.mm(z, z.t()) / temperature)
    
    # Create mask to ignore self-similarity
    batch_size = z.shape[0]
    mask = torch.eye(batch_size, device=device).bool()
    sim = sim.masked_fill(mask, 0)
    
    # Positive pairs are (i, i+N) and (i+N, i)
    # We assume z is ordered as [view1_1, view1_2, ..., view1_N, view2_1, view2_2, ..., view2_N]
    # Wait, usually it's interleaved or stacked. 
    # Let's assume standard implementation where batch is [v1_img1, v2_img1, v1_img2, v2_img2...] or [v1_all, v2_all]
    # Solo implementation does:
    # indexes = batch[0]
    # indexes = indexes.repeat(n_augs)
    # So if batch is N, and n_augs is 2. indexes is 2*N.
    # If we stack outputs: z = torch.cat(out["z"])
    # Then z is [v1_1, ..., v1_N, v2_1, ..., v2_N]
    
    # Let's stick to the [v1_all, v2_all] convention which is easier to implement without indexes if we know batch size.
    N = batch_size // 2
    
    # Positives:
    # For i in 0..N-1 (view 1), positive is i+N (view 2)
    # For i in N..2N-1 (view 2), positive is i-N (view 1)
    
    # We can construct a positive mask
    # The solo implementation uses indexes to be robust to arbitrary batch construction.
    # Here we can simplify.
    
    # sim[i, j] is similarity between i and j.
    # Numerator for i: sim[i, pair(i)]
    # Denominator for i: sum(sim[i, :])
    
    # Pair indices
    # 0 -> N
    # 1 -> N+1
    # ...
    # N -> 0
    # N+1 -> 1
    
    pos_idx = torch.arange(batch_size, device=device)
    pos_idx = (pos_idx + N) % batch_size
    
    # Extract positive similarities
    pos_sim = sim[torch.arange(batch_size), pos_idx]
    
    # Sum of all similarities (excluding self, which is 0)
    den_sim = torch.sum(sim, dim=1)
    
    loss = -torch.log(pos_sim / den_sim)
    return loss.mean()


class SimCLR(nn.Module):
    def __init__(self, backbone_name, proj_hidden_dim, proj_output_dim, cifar=True):
        super().__init__()
        
        if backbone_name == "resnet18":
            self.backbone = resnet18()
            self.features_dim = 512
        elif backbone_name == "resnet50":
            self.backbone = resnet50()
            self.features_dim = 2048
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")

        # Modify backbone for CIFAR
        if cifar:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.backbone.maxpool = nn.Identity()
            
        self.backbone.fc = nn.Identity()

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return {"feats": h, "z": z}
