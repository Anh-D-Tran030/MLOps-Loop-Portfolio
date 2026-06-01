from setuptools import setup, find_packages

setup(
    name="mlops-eval",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["mlops-eval=evaluator.cli:cli"],
    },
    install_requires=[
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "ragas==0.1.21",
        "click>=8.1.0",
    ],
    python_requires=">=3.11",
)
