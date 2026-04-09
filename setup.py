from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("PALACE/palace/palace.py", language_level="3"),
)