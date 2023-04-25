from setuptools import setup

setup(
    name='hmctrack',
    version='0.1',
    packages=['hmctrack'],
    install_requires=['numpy >= 1.19.5',
                      'matplotlib>=3.1.2',
                      'scipy>=1.4.1',
                      'cvxpy>=1.1.2',
                      'torch>=1.10.2',
                      'gpytorch>=1.6.0',
                      'botorch>=0.2.1',
                      'scikit-learn>=0.24.2',
                      'scikit-image>=0.17.2'
                      ]
)