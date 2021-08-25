from setuptools import setup
from version import get_git_version

setup(name='mnc-python',
      version=get_git_version(),
      url='http://github.com/ovro-lwa/mnc-python',
      requirements=['astropy'],
      packages=['mnc'],
      zip_safe=False)
