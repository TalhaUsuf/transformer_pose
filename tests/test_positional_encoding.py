import matplotlib.pyplot as plt
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from model import get_encodings


def test_positional_encoding():
    """
    test case to check the positonal encodings

    Returns
    -------
    True if the test passes
    """
    batch = 10
    seq = 30
    D = 128
    x = torch.randn([batch, seq, D])
    encodings = get_encodings("encoding_only", D)

    encodings_summed = get_encodings("summed_encoding", D)
    out = encodings(x)
    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(out.detach().numpy()[0, :, :])
    ax.set_title("Positional Encoding")
    ax.set_xlabel("Sequence")
    ax.set_ylabel(f"Encoding-Dim.{D} values")
    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=300)
    out_summed = encodings_summed(x)
    assert out.shape == x.shape
    assert out_summed.shape == x.shape
    assert isinstance(encodings, PositionalEncoding1D), "encodings should be an instance of PositionalEncoding1D"
    assert isinstance(encodings_summed, Summer), "encodings_summed should be an instance of Summer"
