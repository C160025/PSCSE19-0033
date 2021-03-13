__version__ = '0.5'

def pretty_versions():
    import numpy
    import cv2
    import scipy
    n_version = numpy.__version__
    c_version = cv2.__version__
    s_version = scipy.__version__
    return "color_transfer : {}, numpy : {} , opencv-python : {} , scipy : {}".format(__version__, n_version, c_version, s_version)