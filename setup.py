from setuptools import setup, find_packages
  
setup(
    name='EBC',
    version='1.0',
    description='Energy Based Clustering',
    author='Moritz Thürlemann',
    author_email='moritz.thuerlemann@phys.chem.ethz.ch',
    packages=find_packages(),    
    install_requires=["scipy>=1.9.0", "scikit-learn>=1.0.1", "networkx>=3.0", "matplotlib>=3.5.3", "numpy>=1.19.5"]
)