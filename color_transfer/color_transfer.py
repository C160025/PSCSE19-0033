'''
instantiate the class with some model and all the necessary function argument,
example like shape or size or background colour,

ColXfer
'''

from __future__ import print_function
from keras_vggface.models import RESNET50, VGG16, SENET50

def ColXfer(include_top=True, model='vgg16', weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=None):
    return null