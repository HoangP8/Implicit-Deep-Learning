from setuptools import setup, find_packages


if __name__ == '__main__':
    # Read in README.md for our long_description
    with open('./README.md', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name='idl',
        version='0.0.1',
        license='MIT',
        description='Official standard library for Implicit Deep Learning Models including State-driven Implicit Models',
        url='https://github.com/HoangP8/Implicit-Deep-Learning',
        package_dir={'': 'src'},
        packages=find_packages("src"),
        package_data={'idl': ['py.typed']},
        include_package_data=True,
        python_requires='>=3.8',
        long_description=long_description,
        long_description_content_type='text/markdown',
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            ],
        install_requires=[
            'torch>=1.11.0',
            'numpy>=1.21.5'
            ],
        author=['Hoang Phan', 'Bao Tran', 'Laurent El Ghaoui'],
        author_email=['21hoang.p@vinuni.edu.vn', '21bao.tq@vinuni.edu.vn', 'laurent.eg@vinuni.edu.vn'],
    )