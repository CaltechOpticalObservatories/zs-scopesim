from setuptools import Extension, setup
from Cython.Build import cythonize


extensions = [
    Extension("palace.palace", ["palace/palace.py"]),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
