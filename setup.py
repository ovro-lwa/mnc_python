from setuptools import setup
from version import get_git_version
from setuptools_scm import get_version

setup(name='mnc-python',
      version=get_version(),
      url='http://github.com/ovro-lwa/mnc-python',
      requirements=['astropy', 'progressbar', 'myst-parser', 'setuptools_scm'],
      packages=['mnc'],
      zip_safe=False)
