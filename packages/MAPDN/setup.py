from setuptools import find_packages, setup

setup(
    name="mapdn",
    version="1.0.0",
    author="PKU-MARL",
    description="PyTorch implementation of HARL Algorithms",
    url="https://github.com/PKU-MARL/HARL",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)