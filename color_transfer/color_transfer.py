from color_transfer.models import Mean_CX, MKL_CX, IDT_CX, REGRAIN_CX

def ColourXfer(source_rgb, target_rgb, model, conversion=None):
    """
    Colour transfer the source image with the target image's colour characteristics.
    :param source_rgb: source image in RGB colour space (0-255) on numpy array uint8
    :param target_rgb: target image in RGB colour space (0-255) on numpy array uint8
    :param conversion: 3 type colour space conversions required for mean model only
                  'opencv' = opencv-python package colour space conversion
                  'matrix' = colour space conversion matrix by Reinhard et al.
                             http://erikreinhard.com/papers/colourtransfer.pdf
                  'noconv' = no colour space conversion
    :param model: 4 type colour transfer models
                  'mean' = Mean model by Reinhard et al.
                           http://erikreinhard.com/papers/colourtransfer.pdf
                  'idt' = IDT model by Pitié et al.
                          https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
                  'regrain' = Regrain model by Pitié et al.
                              https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
                  'mkl' = MKL model by Pitié et al.
                          https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cvmp.pdf
    :return: output_rgb: corrected image in RGB colour space (0-255) on numpy array uint8
    """
    if model == 'mean':
        return Mean_CX(source_rgb, target_rgb, conversion)
    if model == 'idt':
        return IDT_CX(source_rgb, target_rgb)
    if model == 'regrain':
        return REGRAIN_CX(source_rgb, target_rgb)
    if model == 'mkl':
        return MKL_CX(source_rgb, target_rgb)