from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='AutoMxL',
    # Version à modifier également dans AutoMxL.__init__
    version='1.0.0',
    description="Classification Automated Machine Learning (AutoML)",
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/Maxence-Labesse/AutoMxL',
    author='Maxence LABESSE',
    author_email="maxence.labesse@yahoo.fr",
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["AutoMxL", "AutoMxL.Explore", "AutoMxL.Modelisation", "AutoMxL.Preprocessing",
              "AutoMxL.Select_Features", "AutoMxL.Start", "AutoMxL.Utils"],
    include_package_data=True, install_requires=['pandas', 'numpy', 'scikit-learn', 'xgboost', 'numpy', 'torch', 'matplotlib']
)
