""" The purpose of this package is to provide a Python library that automate
the different steps of a ML classification project.
This package contains following sections :
- Load
- Audit
- Preproessing (clearn and process)
- Modelisation

It can be use aswell as a catalog of functions to ease and speed up repetitive 
Data Scientists tasks

"""
from __future__ import absolute_import

# Version of the package
__version__ = "1.0.0"

from . import Load
from . import Audit
from . import Preprocessing
from . import Modelisation
from . import Utils
from . import MLBG59
from .MLBG59 import AutoML
from .Load.Load import load_data
from .Utils.Utils import parse_target


