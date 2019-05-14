import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cae",
    version="0.1",
    author="Arsenii Astashkin",
    author_email="ars.astashkin@gmail.com",
    description="Hybrid Singular Value Decomposition (SVD) implementation",
    long_description=long_description,
    url="https://github.com/arsast/cae",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)