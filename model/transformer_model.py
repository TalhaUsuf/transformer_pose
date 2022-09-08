import torch


class transformer_model(torch.nn.Module):
    """
    Transformer model. to take 128 vector points of pose and detect user fall given
    a batch of videos
    """
