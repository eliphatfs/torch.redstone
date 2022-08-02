import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


def packages():
    return ['torch.' + pkg for pkg in setuptools.find_packages('torch')]


setuptools.setup(
    name="torch-redstone",
    version="0.0.1",
    author="flandre.info",
    author_email="flandre@scarletx.cn",
    description="Redstone torch, common boilerplates for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.6',
)
