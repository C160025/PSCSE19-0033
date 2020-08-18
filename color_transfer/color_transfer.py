from color_transfer.models import Mean_CX, MKL_CX, IDT_CX, REGRAIN_CX

def ColourXfer(source_rgb, target_rgb, model, conversion=None):
    """
    Colour transfer from target image's colour characteristics into source image,
    by the selection colour space conversion model.
    :param source_rgb: source image in RGB colour space (0-255) on numpy array uint8
    :param target_rgb: target image in RGB colour space (0-255) on numpy array uint8
    :param conversion: two type colour space conversions
                  'opencv' = opencv-python package
                  'matrix' = colour space conversion referencing from Reinhard et al. 2001 Color Transfer between Images
                             http://erikreinhard.com/papers/colourtransfer.pdf
    :param model: two type conversion models
                  'mean' = mean and standard deviation transfer referencing from Reinhard et al. 2001 Color Transfer between Images
                           http://erikreinhard.com/papers/colourtransfer.pdf
                  'idt' = probability density function transfer referencing from [Pitie05a], [Pitie05b] and [Pitie07a]
                  'regrain' = regain colour transfer on IDT result referencing from [Pitie05b] and [Pitie07a]
                  [Pitie05a] Pitié et al. 2005 N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
                  [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
                  [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
                  'mkl' = Monge-Kantorovitch linear transfer referencing from
                  [Pitie07b] Pitie et al. 2007 The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer
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