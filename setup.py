#!/usr/bin/env python

import glob
import io
import os

import setuptools


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ).read()


setuptools.setup(
    name="brain-atlas",
    version="0.1",
    license="MIT License",
    description="Scripts for analyzing mouse brain atlas data",
    long_description=read("README.md"),
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py")
    ],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "Click",
        "PyYAML",
        "dask",
        "dask-ml",
        "distributed",
        "google-api-python-client",
        "google-auth",
        "google-cloud-secret-manager",
        "leidenalg",
        "matplotlib",
        "networkx",
        "numpy",
        "openpyxl",
        "pandas",
        "pynndescent",
        "python-igraph",
        "scikit-learn",
        "scipy",
        "sparse",
        "tables",
        "tqdm",
        "umap-learn",
        "zarr",
    ],
    extras_require={"dev": ["black", "isort", "flake8", "pre-commit"]},
    entry_points={"console_scripts": ["atlas = brain_atlas.scripts.__init__:cli"]},
)
