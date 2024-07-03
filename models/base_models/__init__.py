from .cnn import cnn_simple, cnn_bottleneck_simple
from .convnextv2 import convnextv2_atto
from .eva import Eva, eva02_small_patch14_224, eva02_tiny_patch14_224, eva02_base_patch14_224
from .mlp import MLP
from .resnet_dilated import resnet_dilated
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from .swin_transformer_v2 import SwinTransformerV2
from .transformer import vit_tiny_patch2_224, vit_base_patch4_224, vit_small_patch4_224
from .disc_encoder import DiscEncoder