import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()
with open("torch/redstone/version.py", "r") as fh:
    exec(fh.read())
    __version__: str


def packages():
    return ['torch.' + pkg for pkg in setuptools.find_packages('torch')]


setuptools.setup(
    name="torch.redstone",
    version=__version__,
    author="flandre.info",
    author_email="flandre@scarletx.cn",
    description="Redstone torch, common boilerplates for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliphatfs/torch.redstone",
    packages=packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tqdm', 'numpy'
    ],
    python_requires='~=3.6',
)
