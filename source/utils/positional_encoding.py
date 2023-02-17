"""
Source: https://github.com/wzlxjtu/PositionalEncoding2D
"""
import torch
import math


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def positional_encoding_2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from einops import rearrange
    n_h, n_w = 14, 10
    d = 96
    pe = positional_encoding_2d(96, n_h, n_w)
    pe = rearrange(pe, 'd h w -> (h w) d')
    cosine_similarity = torch.einsum('i k, j k -> i j', pe, pe)

    # Display the position encoding
    fig, axes = plt.subplots(n_h, n_w, figsize=(2 * n_w, 2 * n_h))
    for i in range(n_h):
        for j in range(n_w):
            image = rearrange(cosine_similarity[i * n_w + j, :], '(h w) -> h w', h=n_h, w=n_w)
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig('../../visualizations/det_positional_encoding.jpg')
    plt.show()
