# Color Transfer between Images (PSCSE19-0033)
Color Transfer Between Images 

### Way of approach the FYP
- built an open source similar like [VGGface](https://github.com/rcmalli/keras-vggface) 
- explore other [opencv](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) library for converting between color space with D65 color temperature.
- must install all the necessary library within setup.py
- [color transfer](https://github.com/jrosebr1/color_transfer) github reference 

~~~bash
# Most Recent One (Suggested)
pip install git+https://github.com/C160025/PSCSE19-0033
# Release Version
pip install color_transfer
~~~

#### Available Models
```python
import cv2
from color_transfer.color_transfer import ColourXfer

# mean and standard deviation transfer
transfer_rgb = ColourXfer(source_rgb, target_rgb, model='mean', conversion='opencv')
transfer_rgb = ColourXfer(source_rgb, target_rgb, model='mean', conversion='matrix')

# probability density function (pdf) or iterative distribution transfer (idt)
transfer_rgb = ColourXfer(source_rgb, target_rgb, model='idt')

# regain colour transfer
transfer_rgb7 = ColourXfer(source_rgb, target_rgb, model='regrain')

# monge-kantorovitch linear transfer (mkl)
transfer_rgb8 = ColourXfer(source_rgb, target_rgb, model='mkl')

```

### Library Versions
- numpy>=1.18.3
- scipy>=1.4.1
- opencv-python>=4.2.0.34

### References
- [Reinhard et al. 2001 Color Transfer between Images paper](http://erikreinhard.com/papers/colourtransfer.pdf)
- [Pitié et al. 2005 N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer paper](https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf)
- [Pitié et al. 2005 Towards Automated Colour Grading](https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf)
- [Pitié et al. 2007 Automated colour grading using colour distribution transfer](https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf)
- [Pitie et al. 2007 The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer](https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cvmp.pdf)