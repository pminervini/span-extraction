# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
    setup_requires = f.readlines()


setup(name='span-extraction',
      version='0.1',
      description='Span Extraction',
      author='Pasquale Minervini',
      author_email='p.minervini@ucl.ac.uk',
      url='https://github.com/pminervini/span-extraction',
      test_suite='tests',
      license='MIT',
      install_requires=setup_requires,
      setup_requires=setup_requires,
      tests_require=setup_requires,
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages(),
      keywords='span extraction')
