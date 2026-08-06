import os
import sys

from setuptools import setup

# Ensure version.py is importable during PEP 517 isolated builds.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import get_git_version

setup(name='mnc-python',
      version=get_git_version(),
      url='http://github.com/ovro-lwa/mnc-python',
      install_requires=['astropy',
                        'progressbar',
                        'myst-parser',
                        'markdown<3.4'],
      packages=['mnc'],
      py_modules=['version'],
      entry_points='''
        [console_scripts]
        lwamnc=mnc.cli:cli
      ''',
      zip_safe=False)
