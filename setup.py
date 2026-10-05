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
        "scripts/inference.py",
    ],
    install_requires=DEPENDENCIES,
    entry_points={
        "console_scripts": [
            "pvad-prepare-dataset=scripts.prepare_dataset:main",
            "pvad-train=scripts.train:main",
            "pvad-evaluate=scripts.evaluate:main",
            "pvad-export-tflite=scripts.export_tflite:main",
            "pvad-inference=scripts.inference:main",
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
