from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="wmguy",
    version="0.1",
    packages=find_namespace_packages(include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    author="PooNg1223",
    description="Personal AI Assistant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
) 