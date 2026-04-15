from setuptools import setup, find_packages

setup(
    name="pdm_rig_failure",
    version="1.0.0",
    description="Predictive Maintenance for Oil & Gas Drilling Rigs",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "xgboost>=1.7.0",
        "mlflow>=2.5.0",
        "fastapi>=0.100.0",
        "streamlit>=1.25.0",
        "shap>=0.42.0",
    ],
)
