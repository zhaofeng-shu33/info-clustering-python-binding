# pylint: disable=missing-docstring
import setuptools

with open('README.md') as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name='info_cluster',
    version='0.9.post2',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scikit-learn', 'ete3', 'networkx', 'pspartition'],
    author="zhaofeng-shu33",
    author_email="616545598@qq.com",
    description="a hierarchical clustering algorithm based on information theory",
    url="https://github.com/zhaofeng-shu33/info-clustering-python-binding",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    license="Apache License Version 2.0",
)
