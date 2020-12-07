from distutils.core import setup
from setuptools import find_packages

install_reqs = []
install_reqs.extend(["torch ~= 1.7.0", "torchvision ~= 0.8.1", "librl == 0.2.7", "pytest ~= 6.1"])
setup(name='nasrl',
      version='0.2.3',
      description='Library code for our NAS-RL project.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      install_reqs=install_reqs,
      python_requires='~= 3.8',
      packages=find_packages('src'),
      package_dir={'': 'src'},
     )