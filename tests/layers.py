import torch
from simcspr.layers import PerFeatLinear
from simcspr.layers import DecoupledLinear


def test_featlinear():
    ins_features = [3, 4, 5]
    out_features = 2
    model = PerFeatLinear(ins_features, out_features)
    feats_per_space = [torch.randn(10, d) for d in ins_features]
    out = model(feats_per_space)
    print(model)
    s0_z = model.linears[0](feats_per_space[0])
    s1_z = model.linears[1](feats_per_space[1])
    assert torch.allclose(out[:, 0], s0_z)
    assert torch.allclose(out[:, 1], s1_z)
    print(out.shape)
    assert out.shape == (10, 3, 2)
    softmx = torch.softmax(out, dim=-1)
    assert torch.allclose(softmx.sum(dim=-1), torch.ones(10, 3))
    print(softmx.mean(dim=1))
    assert torch.all(softmx.mean(dim=1) < 1.0)
    print("Test passed")


def test_decoupledlinear():
    in_features = 4
    out_features = 3
    num_spaces = 2
    model = DecoupledLinear(in_features, out_features, num_spaces)
    input = torch.randn(10, num_spaces, in_features)
    out = model(input)
    print(model)
    assert out.shape == (10, num_spaces, out_features)
    print("Test passed")


test_decoupledlinear()
