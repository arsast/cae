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
    ],
    license = "MIT",
    install_requires = [
        "joblib == 0.13.2",
        "numpy == 1.16.3",
        "scikit - learn == 0.21.1",
        "scikit - sparse == 0.4.4",
        "scipy == 1.2.1",
        "sklearn == 0.0"
    ]
)
