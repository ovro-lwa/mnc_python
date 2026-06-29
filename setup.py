from setuptools import setup
from version import get_git_version

setup(name='mnc-python',
      version=get_git_version(),
      url='http://github.com/ovro-lwa/mnc-python',
      install_requires=['astropy',
                        'progressbar',
                        'myst-parser',
                        'markdown<3.4'],
      packages=['mnc'],
      entry_points='''
        [console_scripts]
        lwamnc=mnc.cli:cli
      ''',
      zip_safe=False)
