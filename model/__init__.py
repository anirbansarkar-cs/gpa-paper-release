# Core model architectures
from .transformer import TransformerModel
from .cnn import ConvolutionalModel
from .layers import *

# Note: For new code, use the dataset factory:
# from utils.dataset_factory import create_model
# model = create_model('deepstarr', config, 'transformer')
