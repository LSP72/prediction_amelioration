from setuptools import setup, find_packages

setup(
    name="amelio_cp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "scikit-optimize",
    ],
    python_requires=">=3.8",
)
