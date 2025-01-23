from setuptools import setup, find_packages

setup(
    name="mr_gemini",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[
        "google-generativeai>=0.3.0"
    ],
)
