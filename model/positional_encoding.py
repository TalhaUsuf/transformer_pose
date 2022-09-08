from typing import Union

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


def get_encodings(enc_type: str, d_model: int) -> Union[PositionalEncoding1D, Summer]:
    """
    Returns the positional encoding function for the given type and d_model
    it can return the `encoding` or the `vector + encoding(vector)`

    Parameters
    ----------
    enc_type : str
        The type of encoding to return, can be `encoding_only` or `summed_encoding`
    d_model : int
        dimension of the actual vector which needs to be positionally encoded
    Returns
    -------

    """
    if enc_type == "encoding_only":
        # Returns the position encoding only
        encoding = PositionalEncoding1D(d_model)
        return encoding
    if enc_type == "summed_encoding":
        # Return the inputs with the position encoding added
        encoding = Summer(PositionalEncoding1D(d_model))
        return encoding
