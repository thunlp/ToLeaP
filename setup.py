from setuptools import setup, find_namespace_packages

setup(
    name="BodhiAgent",
    version="0.1.0",
    author="Haotian Chen",
    author_email="htchen@tsinghua.edu.cn",
    description="An agent capable of learning how to learn",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hytn/BodhiAgent",
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md'],
    },
    license="MIT",
)
