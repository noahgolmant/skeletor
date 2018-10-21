"""setup.py for skeletor """

from setuptools import setup

install_requires = [
    'numpy>=0.14',
    'matplotlib>=2.2.2',
    'scikit-learn>=0.19.1',
    'pylint>=1.9.1',
    'yapf>=0.22.0',
    'pyyaml>=3.12',
    'ray>=0.4.0',
    'scipy>=1.1.0',
    'python-dotenv',
    'track',
    'awscli'
]

dependency_links = [
    'git+https://github.com/richardliaw/track/tarball/master#egg=track'
]

setup(name="skeletor", author="noahgolmant",
      install_requires=install_requires,
      dependency_links=dependency_links)
