# -*- coding: utf-8 -*-

"""
Set-up
-----------------
"""

from distutils.core import setup


setup(
    name='mentoria-ayc',
    version='1.0',
    description='Analisis y Curacion de datos',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'jupyterlab',
        'scikit-learn',
        'ftfy<5.6',
    ]
)
