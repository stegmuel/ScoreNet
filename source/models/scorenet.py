from source.patch_selection.perturbed_topk import PerturbedTopK
import source.models.vision_transformers_peg as vits
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import torch
import math


class ScoreNet(nn.Module):
    def __init__(self, args):
        super(ScoreNet, self).__init__()
        # Misc. attributes
        self.scale = args.scale
        self.n_cls_patch = args.n_cls_patch
        self.n_cls_scorer = args.n_cls_scorer
        self.large_patch_size = args.large_patch_size

        self.patch_model = vits.__dict__['vit_tiny'](patch_size=args.small_patch_size)
        self.perturbed_topk = PerturbedTopK(k=args.k, sigma=args.sigma)
        self.scorer_model = vits.__dict__['vit_tiny'](patch_size=args.small_patch_size)
        self.encoder = vits.TransformerEncoder(self.scorer_model.embed_dim)
        self.embed_dim = self.scorer_model.embed_dim * (self.n_cls_scorer + self.n_cls_patch)
        self.ratio = torch.tensor(0.)

        # Initialize the convolution aggregating the self-attentions of different heads
        self.aggregation_conv = nn.Conv2d(
            in_channels=self.scorer_model.num_heads,
            out_channels=1,
            kernel_size=(1, 1),
            padding=(0, 0),
            bias=False
        )

    def get_saliency_maps(self, thumbnails):
        _, attention, _ = self.scorer_model.vit.get_last_selfattention(thumbnails, 1)
        _, _, h, w = thumbnails.shape
        n_h, n_w = h // self.small_patch_size, w // self.small_patch_size
        attention = rearrange(attention[:, :, 0, 1:], 'b c (h w) -> b c h w', h=n_h, w=n_w)

        # Aggregate the self-attentions
        conv_weight = self.aggregation_conv.weight.data
        conv_weight = (conv_weight - conv_weight.min()) / (conv_weight.max() - conv_weight.min())
        conv_weight = conv_weight / conv_weight.sum()
        self.aggregation_conv.weight.data = conv_weight
        attention = self.aggregation_conv(attention)
        return attention.squeeze()

    def compute_average_patch_number(self, indicators):
        """
        Computes the ratio of effectively selected patches and queried patches. It indicates the selectivity of the
        scorer.
        :param indicators: tensor indicating the position and weight of the selected patches.
        :return: None.
        """
        ratio = torch.count_nonzero(indicators, dim=-1).float().mean()
        beta = 0.9
        self.ratio = beta * ratio + (1 - beta) * self.ratio

    def get_indicators(self, images, use_hard_topk):
        # Get the thumbnails
        thumbnails = F.interpolate(images, scale_factor=(1 / self.scale, 1 / self.scale), mode='bicubic')

        # Get the self-attention
        outputs, attentions = self.scorer_model.get_last_selfattention(thumbnails, self.n_cls_scorer)
        outputs = outputs[:, 0, :]
        b, c, h, w = images.shape
        n_h, n_w = h // self.large_patch_size, w // self.large_patch_size
        attentions = rearrange(attentions[:, :, 0, 1:], 'b h (m n) -> b h m n', m=n_h, n=n_w)

        # Aggregate the self attentions of different heads
        conv_weight = self.aggregation_conv.weight
        conv_weight = (conv_weight - conv_weight.min()) / (conv_weight.max() - conv_weight.min())
        conv_weight = conv_weight / conv_weight.sum()
        self.aggregation_conv.weight.data = conv_weight
        scores = self.aggregation_conv(attentions)
        scores = rearrange(scores, 'b 1 h w -> b (h w)')

        # Get the indicators
        n = scores.shape[-1]
        k = math.ceil(0.1 * n)
        indices = torch.topk(scores, k, dim=-1).indices
        hard_indicators = F.one_hot(indices, num_classes=n).float().to(scores.device)
        if use_hard_topk:
            indicators = hard_indicators
        else:
            soft_indicators = self.perturbed_topk(scores, k)
            indicators = soft_indicators - soft_indicators.detach() + hard_indicators
            self.compute_average_patch_number(soft_indicators.detach())
        return indicators, outputs

    def forward(self, x, use_hard_topk):
        # Get the selected patches
        indicators, scorer_cls = self.get_indicators(x, use_hard_topk)

        # Get the patches from the image
        patches = rearrange(x, 'b c (m h) (n w) -> b (m n) c h w', h=self.large_patch_size, w=self.large_patch_size)

        # Re-weight the patches
        patches = torch.einsum('b k n, b n c h w -> b k c h w', indicators, patches)

        # Embed the patches
        patches = rearrange(patches, 'b k c h w -> (b k) c h w')
        patches = self.patch_model(patches)
        patches = rearrange(patches, '(b k) d -> b k d', b=x.shape[0])

        hidden = self.encoder(patches, self.n_cls_patch)
        hidden = torch.cat([scorer_cls, hidden], dim=-1)
        return hidden
