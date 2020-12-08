import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyutil", # Replace with your own username
    version="0.0.2",
    author="Martin Jordal Hovin",
    author_email="martinhovin91@gmail.com",
    description="Quality of life package for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martijho/util",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)