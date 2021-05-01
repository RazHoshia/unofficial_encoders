# Unofficial Encoders
Unofficial but extremely useful Label and One Hot encoders.
Always wanted to be able to use One Hot encoding with string values? to use a label encoder that can handle unknown values? This package finally allows you to do so!
<br>The Encoders are 100% skelarn compatible and were tested with both skelarn compatible tests and hypothesis fuzz testing.
## Installation
```pip install git+https://github.com/RazHoshia/unofficial_encoders.git```

## How To Use
Please check the [examples folder](https://github.com/RazHoshia/unofficial_encoders/tree/main/examples).

## Developers
Please feel free to open issues and pull requests.

### Installation
```
git clone https://github.com/RazHoshia/unofficial_encoders.git
cd unofficial_encoders
pip install -e . # install in editable mode, see https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html
```
### Development
Before Submitting PRs, please check your code using tox. The project's tox.ini can be found [here](https://github.com/RazHoshia/unofficial_encoders/blob/main/tox.ini).
#### How to test
```
pip install tox
cd unofficial_encoders
tox
```
## Refrences
- https://scikit-learn.org/stable/developers/develop.html
- https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
