from setuptools import setup, find_packages

setup(name='src',
      version='0.0.1',
      description='dm2gal',
      author='Noah Kasmanoff',
      author_email='nsk367@nyu.edu',
      url='https://github.com/nkasmanoff/dm2gal',
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages())
