#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup dialtts libarary."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "dialtts requires Python>=3.6, "
        "but your Python is {}".format(sys.version))

requirements = {
    "install": [
        "torch>=2.0",
        "setuptools>=38.5.1",
        "PyYAML>=3.12",
        "numpy",
        "opencc",
    ],
    "setup": [
    ],
    "test": [
    ]
}
entry_points = {
    "console_scripts": [
        "dialtts-acoustic-train=dialtts.acoustic.bin.train:main",
        "dialtts-acoustic-infer=dialtts.acoustic.bin.infer:main",
        "dialtts-acoustic-export=dialtts.acoustic.bin.export:main",
        "dialtts-vocoder-train=dialtts.vocoder.bin.train:main",
        "dialtts-vocoder-infer=dialtts.vocoder.bin.infer:main",
        "dialtts-vocoder-export=dialtts.vocoder.bin.export:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="dialtts",
      version="0.1.0",
      url="http://gitlab.xxx.com/wuwen.yww/DialTTS.git",
      author="wuwen.yww",
      author_email="wuwen.yww@xxx.com",
      description="TouchTTS implementation",
      long_description=open(os.path.join(dirname, "README.md"),
                            encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      license="MIT License",
      packages=find_packages(include=["dialtts*"]),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      entry_points=entry_points,
      classifiers=[
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: MIT License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )
