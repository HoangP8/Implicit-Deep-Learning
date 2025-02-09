# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'idl'
copyright = '2025, Hoang Phan, Bao Tran, Chi Nguyen, Bao Truong, Thanh Tran, Khai Nguyen, Hong Chu, Laurent El Ghaoui'
author = 'Hoang Phan, Bao Tran, Chi Nguyen, Bao Truong, Thanh Tran, Khai Nguyen, Hong Chu, Laurent El Ghaoui'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinxawesome_theme'

# Theme options
html_theme_options = {
    'show_prev_next': True,
    'repository_url': 'https://github.com/HoangP8/Implicit-Deep-Learning',
    'use_repository_button': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

pygments_style = "sphinx" 

