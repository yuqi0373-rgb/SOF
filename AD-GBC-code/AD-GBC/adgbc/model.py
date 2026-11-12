from .rolling import RollingADGBC
from .ukan import UKANADGBC


def build_model(backbone='rolling', num_classes=1):
    if backbone == 'rolling':
        return RollingADGBC(num_classes=num_classes, input_channels=3)
    if backbone == 'ukan':
        return UKANADGBC(num_classes=num_classes, input_channels=3)
    raise ValueError(f'Unsupported backbone: {backbone}')
