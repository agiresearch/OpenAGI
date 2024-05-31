# setup.py
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='openagi',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agiresearch",
    packages=['openagi'],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Machine Learning",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.7, <4",
)