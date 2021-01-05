""" The purpose of this package is to provide a Python library that automate
the different steps of a ML classification project.
This package contains following sections :
- Start
- Explore
- Preprocessing (clear and process)
- Select features
- Modelisation

It can be use as a catalog of functions to ease and speed up repetitive aswell
Data_handling Scientists tasks

"""
from __future__ import absolute_import

# Version of the package (à modifier également dans setup.py
__version__ = "1.0.0"

from . import Start
from . import Explore
from . import Preprocessing
from . import Select_Features
from . import Modelisation
from . import Utils
from . import __main__
from .__main__ import AML
from .Start.Load import *
from .Start.Encode_Target import *
