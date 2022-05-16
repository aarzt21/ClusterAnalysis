from setuptools import setup


CLASSIFIERS = '''\
License :: OSI Approved
Programming Language :: Python :: 3.7 :: 3.8
Topic :: Unsupervised Machine Learning
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
'''

DISTNAME = 'ClusterAnalysis'
AUTHOR = 'Alex Arzt'
AUTHOR_EMAIL = 'ax.arzt@email.com'
DESCRIPTION = 'This is a simple package that allows infer the ideal k (number of clusters) for several clustering algorithms.'
LICENSE = 'MIT'
README = 'This is a simple package that allows infer the ideal k (number of clusters) for several clustering algorithms.'

VERSION = '0.1.0'
ISRELEASED = False

PYTHON_MIN_VERSION = '3.7'
PYTHON_MAX_VERSION = '3.10.1'
PYTHON_REQUIRES = f'>={PYTHON_MIN_VERSION}, <={PYTHON_MAX_VERSION}'

INSTALL_REQUIRES = [
    'numpy',
    'scipy', 
    'matplotlib',
    'scikit-learn',
    'scipy',
    'seaborn',
    'typing',
    'copy',
    'pandas'
]

PACKAGES = [
    'ClusterAnalysis'
]

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
