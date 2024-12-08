from setuptools import setup, find_packages

setup(
    name='svj_tools',  # Name of the package
    version='0.1.0',   # Package version
    packages=find_packages(),  # Automatically finds all packages and subpackages
    install_requires=[
        'numpy',  # List of dependencies
        'matplotlib',
    ],
    author='Roberto Seidita',
    author_email='',
    description='A Python package for GNN-based analysis of Semi-visible Jet data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rseidita/',  # Update with your GitHub repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version
)
