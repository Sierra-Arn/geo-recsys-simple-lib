from setuptools import setup, find_packages

setup(
    name="geo-recsys-simple-lib",
    version="0.1.0",
    author="Sierra Arn",
    description="Simple Geographic Recomendation System Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    python_requires=">=3.13",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn>=1.7.0,<2",
        "pandas>=2.3.0,<3",
        "numpy>=2.3.1,<3",
        "pydantic >=2.11.7,<3",
        "huggingface_hub >=0.33.1,<0.34"
    ]
)
