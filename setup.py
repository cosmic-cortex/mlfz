from setuptools import setup, find_packages


setup(
    name="mlfz",
    version="0.1.3.2",
    author="Tivadar Danka",
    description="Machine Learning From Zero: an educational machine learning library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cosmic-cortex/mlfz",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
