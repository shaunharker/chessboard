import sys

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

###
MODULENAME = "chessboard"
VERSION = "0.0.1"
DESCRIPTION = "A Python chessboard with FEN and legal move generation. Implemented with C++ and pybind11."
###

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(MODULENAME,
        ["src/chessboard.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', VERSION)],
        cxx_std="17"
        ),
]

setup(
    name=MODULENAME,
    version=VERSION,
    author='Shaun Harker',
    author_email='sharker81@gmail.com',
    url = f'https://github.com/shaunharker/{MODULENAME}',
    description=DESCRIPTION,
    long_description='',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    download_url = '',
    install_requires=[]
)
