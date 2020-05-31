from setuptools import setup
from setuptools import find_packages

setup(name='mrtool',
      version='0.0.1',
      description='Meta-regression tool',
      url='https://github.com/ihmeuw-msca/SFMA',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest',
                        'ipopt',
                        'limetr',
                        'xspline',
                        'pycddlib'],
      zip_safe=False)