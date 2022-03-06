from setuptools import setup

setup(
    name='oscina',
    version='0.0.0',
    description='OscInA: tools to test for oscillations in autocorrelated signals',
    url='https://github.com/gbrookshire/oscina',
    author='Geoffrey Brookshire',
    packages=['oscina'],
    install_requires=[
        'tqdm',
        'numpy>=1.18.1',
        'sklearn',
    ],
    zip_safe=False
)
