from setuptools import setup, find_packages
exec(open('colour_transfer/version.py').read())
setup(
    name='colour_transfer',
    version=__version__,
    description='colour transfer between images implementation',
    url='https://github.com/C160025/PSCSE19-0033',
    author='Low Huang Hock',
    author_email="c160025@e.ntu.edu.sg",
    keywords=[
        'color transfer between images', 'colour grading', 'N-dimensional pdf matching', 'iterative distribution transfer',
        'reduce grain noise artefact', 'Monge-Kantorovitch linear transfer'
    ],
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy>=1.18.3', 'scipy>=1.4.1', 'opencv-python>=4.2.0.34'
    ])