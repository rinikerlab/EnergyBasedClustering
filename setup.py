from setuptools import setup, find_packages
  
setup(
    name='EBC',
    version='0.9',
    description='Energy Based Clustering',
    author='Moritz Thürlemann',
    author_email='moritz.thuerlemann@phys.chem.ethz.ch',
    packages=find_packages(include=['EBC']),
    package_dir={'':'source'},
    install_requires=["scipy>=1.9.0", "scikit-learn>=1.0.1", "networkx>=2.8.8", "matplotlib>=3.5.3", "numpy>=1.19.5"]
)