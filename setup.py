from setuptools import setup

setup(name='mrtool',
      version='0.0.0',
      description='Meta-regression tool',
      url='https://github.com/ihmeuw-msca/SFMA',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=['mrtool'],
      package_dir={'': 'src'},
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest',
                        'ipopt',
                        'limetr',
                        'xspline'],
      zip_safe=False)