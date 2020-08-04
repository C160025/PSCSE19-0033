from color_transfer.models import Mean_CX, MKL_CX, IDT_CX, REGRAIN_CX

def ColourXfer(source_rgb, target_rgb, model, conversion=None):
    """
    Colour transfer from target image's colour characteristics into source image,
    by the selection colour space conversion model.
    :param source_rgb: source image in RGB colour space (0-255) on numpy array uint8
    :param target_rgb: target image in RGB colour space (0-255) on numpy array uint8
    :param conversion: two type colour space conversions
                  'opencv' = opencv-python package
                  'matrix' = equation referencing from Color Transfer between Images by Erik Reinhard's paper
                             http://erikreinhard.com/papers/colourtransfer.pdf
    :param model: two type conversion models
                  'mean' = compute using mean and standard deviation referencing from
                           Color Transfer between Images by Erik Reinhard's paper
                           http://erikreinhard.com/papers/colourtransfer.pdf
                  'pdf' = compute using
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
                  'idt' = compute using
                  Automated Colour Grading using Colour Distribution Transfer by F. Pitié
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
                  'mkl' = compute using
                  The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer by F. Pitié
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