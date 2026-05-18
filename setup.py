from setuptools import find_packages, setup

setup(
    name="darp_nego",
    version="0.0.1",
    author="Ivan Gimenez Palacios, Juan Miguel Alberola, and Víctor Sánchez Anguix",
    description="Negotiation models for DARP",
    license="MIT",
    packages=find_packages(include=["darp_nego", "darp_nego.*"]),
)
