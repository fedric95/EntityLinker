from setuptools import setup

setup(
    name='EntityLinker',
    version='0.1.0',    
    packages=['EntityLinker'],
    install_requires=[
        'spacy>=2.3.4',
        'sentence_transformers>=0.4.1'
    ]
)
