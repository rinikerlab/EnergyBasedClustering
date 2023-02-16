from setuptools import setup, find_packages
  
setup(
    name='EBC',
    version='0.9',
    description='Energy Based Clustering',
    author='Moritz Thürlemann',
    author_email='moritz.thuerlemann@phys.chem.ethz.ch',
    packages=find_packages(include=['EBC']),
    package_dir={'':'source'}
)