# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import builtins

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..')
SPHINX_MOCK_REQUIREMENTS = int(os.environ.get('SPHINX_MOCK_REQUIREMENTS', True))
builtins.__EXPERIMENTING_SETUP__ = True
import experimenting  # noqa: E402

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'm2r']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tools']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'  #

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Other
def package_list_from_file(file):
    mocked_packages = []
    with open(file, 'r') as fp:
        for ln in fp.readlines():
            found = [ln.index(ch) for ch in list(',=<>#') if ch in ln]
            pkg = ln[: min(found)] if found else ln
            if "git+" in pkg:
                continue
            if pkg.rstrip():
                mocked_packages.append(pkg.rstrip())
    return mocked_packages


MOCK_PACKAGES = []
if SPHINX_MOCK_REQUIREMENTS:
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += package_list_from_file(os.path.join(PATH_ROOT, 'requirements.txt'))

MOCK_MANUAL_PACKAGES = [
    # packages with different package name compare to import name
    'hydra',
    'cv2',
    'pytorch_lightning',
    'pose3d_utils',
    
]
autodoc_mock_imports = MOCK_PACKAGES + MOCK_MANUAL_PACKAGES

# -- Project information -----------------------------------------------------

project = '2021 CVPRw Lifting Monocular Events to 3D Human Poses'
copyright = experimenting.__copyright__
author = experimenting.__author__
version = experimenting.__version__
