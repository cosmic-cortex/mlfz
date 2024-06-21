from setuptools import setup, find_packages


setup(
    name="mlfz",
    version="0.0.0",
    author="Tivadar Danka",
    description="An educational machine learning library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_pypi_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
