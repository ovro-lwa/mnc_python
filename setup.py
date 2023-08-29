from setuptools import setup
from setuptools_scm import get_version

setup(name='mnc-python',
      version=get_version(),
      url='http://github.com/ovro-lwa/mnc-python',
      install_requires=['astropy',
                        'progressbar',
                        'myst-parser',
                        'setuptools_scm'],
      packages=['mnc'],
      entry_points='''
        [console_scripts]
        lwamnc=mnc.cli:cli
      ''',
      zip_safe=False)
