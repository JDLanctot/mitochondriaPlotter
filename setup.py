#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
from distutils.core import setup

pkg_name = 'mitochondriaPlotter'
author = "Jordan D. Lanctot"
author_email = "jordan.lanctot@torontomu.ca"

install_requires = ['abstractcp',
                    'matplotlib',
                    'networkx',
                    'numpy',
                    'more-itertools',
                    'pyyaml',
                    'scipy',
                    'tqdm']

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        description="Mitochondria Plotter",
        author=author,
        author_email=author_email,
        packages=setuptools.find_packages(),
        python_requires='>=3.8',
        install_requires=install_requires
)

