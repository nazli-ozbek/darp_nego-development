from setuptools import setup

setup(
    name="darp_nego",
    version="0.0.1",
    author="Ivan Gimenez Palacios, Juan Miguel Alberola, and Víctor Sánchez Anguix",
    description="Negotiation models for DARP",
    license="MIT",
    packages=["darp_nego", "darp_nego/utils", "darp_nego/darp", "darp_nego/negotiator", "darp_nego/protocol", "darp_nego/outcome", "darp_nego/test"]
)
