"""
this script defines transformer model. implementation is the same as of 2017 Vaswani paper.
this uses pytorch implementation as the base.
"""

import torch

from .positional_encoding import get_encodings


class transformer_model(torch.nn.Module):
    """
    Transformer model. to take 128 vector points of pose and detect user fall given
    a batch of videos
    """

    def __init__(self, d_model: int, nhead_attention: int, dim_feedforward: int, dropout: float = 0.25, activation: str = "relu", encoder_layers: int = 6):
        """
        initializer of the transformer model

        Parameters
        ----------
        d_model : int
        nhead_attention : int
        dim_feedforward : int
        dropout : float, optional
        activation : str, optional
        encoder_layers : int, optional
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead_attention
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = encoder_layers
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation, batch_first=True)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers, norm=None)

        # positonal encodings layer
        self.pe = get_encodings("summed_encoding", self.d_model)
        #     linear layer which will classify if the user has fallen or not
        self.classifier = torch.nn.Linear(self.d_model, 1)

    def forward(self, x):
        """
        forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            shape must be (batch, seq, d_model)

        Returns
        -------
        torch.Tensor
            shape is (batch, seq, d_model)
        """
        # add Positional encodings to pose vector
        x = self.pe(x)
        x = self.encoder(x)  # [batch, seq, d_model]
        # sum along sequence dimension
        x = torch.sum(x, dim=1)  # [batch, d_model]
        x = self.classifier(x)  # [batch, 1]
        return x
