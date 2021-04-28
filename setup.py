import setuptools

setuptools.setup(
    name="unofficial_encoders",
    version="0.0.1.dev",
    author="Raz Hoshia",
    author_email="razhoshia@gmail.com",
    description="unofficial but useful sklearn compatible encoders",
    url="https://github.com/RazHoshia/unofficial_encoders",  # FIXME change
    project_urls={
        "Bug Tracker": "https://github.com/RazHoshia/unofficial_encoders/issues",  # FIXME change
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=('tests')),
    python_requires=">=3.6",
    install_requires=[
        'scikit-learn>0.23.0',
        'pandas'
    ]
)
