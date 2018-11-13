from distutils.core import setup

setup(
    name='NewResNext',
    version='0.1dev',
    description='Implementation of ResNext using Pytorch',
    long_description=open('README.txt').read(),
    packages=['new_resnext',],
    license='GPLv3+',
    author='Anna Fedorova',
    author_email='anyutka-fedorova@mail.ru',
    package_data = {'': ['LICENSE.txt']},
    install_requires=[
        'pytorch >= 0.4.1',
        'tensorboardX >= 1.2',
        'torchvision >= 0.2.1',
        'tensorboard >= 1.8.0',
        'scikit-learn >= 0.19.1',
        'numpy >= 1.14.3'
    ],
)