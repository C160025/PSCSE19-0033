from color_transfer.genr_result import genr_image_result
from color_transfer.genr_result import genr_1d_result
from color_transfer.genr_result import genr_2d_result, genr_idt_iter_result

# image must be in png format
source = "images/exp6_source.png"
target = "images/exp6_target.png"
genr_image_result(source, target)
genr_1d_result(source, target)
genr_2d_result(source, target)
genr_idt_iter_result()
