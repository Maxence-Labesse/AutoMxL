from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE/ "README.md").read_text()

setup(
    name='MLBG59',
    version='1.0.0',
    description = "Classification Automated Machine Learning (AutoML)",
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/Maxence-Labesse/MLBG59',
    author='Maxence LABESSE',
    author_email="maxence.labesse@yahoo.fr",
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT Licence",
        "Programming Language :: Python 3",
        "Programming Language :: Python 3.7"
    ],
    packages=["MLBG59"],
    include_package_data=True, install_requires=['pandas', 'scikit-learn', 'xgboost', 'numpy']
    #install_requires=["XGBOOST"],
)
