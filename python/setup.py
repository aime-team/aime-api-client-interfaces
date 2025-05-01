from setuptools import setup

setup(
    name='aime-api-client-interface',
    version='0.8.9',
    author='AIME',
    author_email='carlo@aime.info',
    packages=['aime_api_client_interface'],
    install_requires=[
        'requests==2.31.0',
        'aiohttp==3.9.0',
    ],
    zip_safe=False
)
