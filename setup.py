import setuptools

setuptools.setup(
    name="unofficial_encoders",
    version="0.0.1.dev",
    author="Raz Hoshia",
    author_email="razhoshia@gmail.com",
    description="unoffical but usefull sklearn compatible encoders",
    url="https://github.com/RazHoshia/MedTPOT", # FIXME change
    project_urls={
        "Bug Tracker": "https://github.com/RazHoshia/MedTPOT/issues", # FIXME change
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