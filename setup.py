from setuptools import setup, find_packages

setup(
    name='coalign',
    version='0.0.0',
    packages=find_packages(),  # This will include 'adatest' if it's structured as a package
    install_requires=[
        'nest_asyncio', 
        'numpy', 
        'aiohttp_security', 
        'pandas',
        # other dependencies
    ],
    # additional metadata
)