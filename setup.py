# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for the Personal VAD (personal_vad) package."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  LONG_DESCRIPTION = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fin:
  DEPENDENCIES = [
      line.strip()
      for line in fin
      if line.strip() and not line.strip().startswith("#")
  ]

setuptools.setup(
    name="personal-vad",
    version="0.1.0",
    author="Quan Wang, Shaojin Ding, Ignacio Lopez Moreno",
    author_email="quanw@google.com",
    description=(
        "Personal VAD: Speaker-Conditioned Target Voice Activity Detection"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/personal_vad",
    packages=setuptools.find_packages(include=["personal_vad", "scripts"]),
    scripts=[
        "scripts/prepare_dataset.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/export_tflite.py",
    ],
    install_requires=DEPENDENCIES,
    entry_points={
        "console_scripts": [
            "pvad-prepare-dataset=scripts.prepare_dataset:main",
            "pvad-train=scripts.train:main",
            "pvad-evaluate=scripts.evaluate:main",
            "pvad-export-tflite=scripts.export_tflite:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
