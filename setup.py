from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE/ "README.md").read_text()

setup(
    name='MLBG59',
    version='1.0.0',
    url='https://github.com/Maxence-Labesse/MLBG59',

    license='',
    author='Maxence LABESSE',
    author_email='maxence.labesse@yahoo.fr',
    description='Classification Automated Machine Learning'
)
