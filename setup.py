
import setuptools
import os

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret


path = os.path.dirname(os.path.abspath(__file__))
requires =  get_requirements(path)
print(requires)

with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'opendelta',
        version = '0.0.1',
        description = "An open source framework for delta learning (parameter efficient learning).",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author = '',
        author_email = 'shengdinghu@gmail.com',
        license="Apache",
        url="https://github.com/thunlp/OpenDelta",
        keywords = ['PLM', 'Parameter-efficient-Learning', 'AI', 'NLP'],
        python_requires=">=3.8.0",
        install_requires=requires,
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ]
    )