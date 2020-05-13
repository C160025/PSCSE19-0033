__version__ = '0.1'

def pretty_versions():
    import numpy
    import cv2
    n_version = numpy.__version__
    c_version = cv2.__version__
    return "color-transfer : {}, numpy : {} , opencv-python : {} ".format(__version__, n_version, c_version)