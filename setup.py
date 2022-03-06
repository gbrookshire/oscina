import os.path
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('oscina', 'version.py')) as version_file:
    exec(version_file.read())

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
    package_data={"project": ["oscina/*.yaml"]},
    zip_safe=False
)
