from setuptools import setup

setup(
    name='AIME API Client Interface',
    version='0.8.8',
    author='AIME',
    author_email='carlo@aime.info',
    packages=['aime_api_client_interface'],
    install_requires=[
        'requests==2.31.0',
        'aiohttp==3.9.0',
    ],
    zip_safe=False
)
