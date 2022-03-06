from setuptools import setup, find_packages
from oscina.version import __version__

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()


setup(
    name='oscina',
    version=__version__,
    description='OscInA: tools to test for oscillations in autocorrelated signals',
    url='https://github.com/gbrookshire/oscina',
    author='Geoffrey Brookshire',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=['unittest'],
    zip_safe=False
)
