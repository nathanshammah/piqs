#!/usr/bin/env python
"""PIQS: Permutational Invariant Quantum Solver

PIQS is an open-source Python solver to study the exact Lindbladian
dynamics of open quantum systems consisting of identical qubits.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""
import os
import sys

# The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    from setuptools import setup, Extension
    TEST_SUITE = 'nose.collector'
    TESTS_REQUIRE = ['nose']
    EXTRA_KWARGS = {
        'test_suite': TEST_SUITE,
        'tests_require': TESTS_REQUIRE
    }
except:
    from distutils.core import setup
    from distutils.extension import Extension
    EXTRA_KWARGS = {}

try:
    import numpy as np
except:
    np = None

from Cython.Build import cythonize
from Cython.Distutils import build_ext

MAJOR = 1
MINOR = 2
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.8)', 'scipy (>=0.15)', 'cython (>=0.21)', 'qutip (>=4.2)']
INSTALL_REQUIRES = ['numpy>=1.8', 'scipy>=0.15', 'cython>=0.21', 'qutip>=4.2']
PACKAGES = ['piqs', 'piqs/cy', 'piqs/tests']
PACKAGE_DATA = {
    'piqs': ['configspec.ini'],
    'piqs/tests': ['*.ini'],
    'piqs/cy': ['*.pxi', '*.pxd', '*.pyx'],
}
INCLUDE_DIRS = [np.get_include()] if np is not None else []
NAME = "piqs"
AUTHOR = ("Nathan Shammah, Shahnawaz Ahmed")
AUTHOR_EMAIL = ("nathan.shammah@gmail.com, shahnawaz.ahmed95@gmail.com")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "quantum physics dynamics permutational symmetry invariance"
URL = ""
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]

# Add Cython extensions here
cy_exts = ['dicke']

# If on Win and Python version >= 3.5 and not in MSYS2 (i.e. Visual studio compile)
if sys.platform == 'win32' and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35 and os.environ.get('MSYSTEM') is None:
    _compiler_flags = ['/w', '/Ox']
# Everything else
else:
    _compiler_flags = ['-w', '-O3', '-march=native', '-funroll-loops']

EXT_MODULES =[]
# Add Cython files from piqs/cy
for ext in cy_exts:
    _mod = Extension('piqs.cy.'+ext,
            sources = ['piqs/cy/'+ext+'.pyx'],
            include_dirs = [np.get_include()],
            extra_compile_args=_compiler_flags,
            extra_link_args=[],
            language='c++')
    EXT_MODULES.append(_mod)

# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

# Setup commands go here
setup(
    name = NAME,
    version = VERSION,
    packages = PACKAGES,
    include_package_data=True,
    include_dirs = INCLUDE_DIRS,
    ext_modules = cythonize(EXT_MODULES),
    cmdclass = {'build_ext': build_ext},
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    license = LICENSE,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    keywords = KEYWORDS,
    url = URL,
    classifiers = CLASSIFIERS,
    platforms = PLATFORMS,
    requires = REQUIRES,
    package_data = PACKAGE_DATA,
    zip_safe = False,
    install_requires=INSTALL_REQUIRES,
    **EXTRA_KWARGS
)
