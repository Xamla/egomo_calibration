#!/usr/bin/env python

from setuptools import setup

setup(name='python',
      version='0.0.1',
      description='Python files for robot calibration',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7 :: 3.5 :: 3.6',
        'Topic :: Robot Calibration',
      ],
      url='http://github.com/xamla',
      author='Inga Altrogge',
      author_email='inga.altrogge@xamla.com',
      license='none',
      packages=['python'],
      install_requires=[
          'numpy',
          'markdown',
      ],
      zip_safe=False)
