from setuptools import setup, find_packages

setup(
    name='activeLearning_with_Fastfit',
    version='0.1.0',
    author='Zijian an',
    author_email='vilasazj@gmail.com',
    packages=find_packages(),
    description='A package for annotating and training models with a MySQL backend.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas',
        'pigeonXT',
        'fastfit',
        'datasets',
        'mysql-connector-python',
        'transformers[torch]',  # Note the [torch] specifier for additional dependencies
        'tqdm',
        'torch',
        'ipykernel',
        'ipywidgets',
        'wandb',
        'pyarrow',
        'jupyter-ui-poll',
        'huggingface_hub'
    ],
    python_requires='>=3.6',
)
