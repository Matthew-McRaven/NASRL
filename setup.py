from distutils.core import setup
from setuptools import find_packages

install_reqs = []
install_reqs.extend(["networkx ~= 2.5", "torch ~= 1.6.0", "torchvision ~= 0.7.0", "librl == 0.2.0"])
setup(name='nasrl',
      version='0.2.1',
      description='Library code for our NAS-RL project.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      install_reqs=install_reqs,
      python_requires='~= 3.8',
      packages=find_packages(),
     )