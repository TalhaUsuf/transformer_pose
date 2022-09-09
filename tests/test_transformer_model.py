import torch

from model import transformer_model


def test_transformer_model():
    Batch = 10
    Seq = 30
    dim = 128
    attention_heads = 8
    trf_model = transformer_model(dim, attention_heads, 512)
    # sample input
    x = torch.randn([Batch, Seq, dim])
    out = trf_model(x)
    assert out.shape == x.shape, f"output shape should be same as input shape, but found {out.shape} and {x.shape}"
    assert out.dtype == x.dtype, f"output dtype should be same as input dtype, but found {out.dtype} and {x.dtype}"
