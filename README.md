# LoonyLander
CSC 480 Final Project

Environment: [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)

## Setup
### System Requirements
- Python 3.3+
- A dependency manager like HomeBrew.
- Pip

### Details
This project uses Python 3.6 with the Anaconda distribution. You may use any version of Python 3.3+, and you must have `pip` installed. If your `pip` is not automatically pointed to Python3, then replace all following commands with `pip3`. You can also install using the appropriate `conda install` command.

### External Dependencies
- OpenAI Gym: `pip install gym`
- Box2d:
  1. `brew install swig` (or the appropriate Linux command).
  2. Somewhere, not necessarily in the project, do `git clone https://github.com/pybox2d/pybox2d pybox2d_dev`
  3. `cd pybox2d_dev/`
  4. `python setup.py build`
  5. `python setup.py install`
  6. `sudo pip install box2d-py`.
- NumPy: `pip install numpy`
- Scipy: `pip install scipy`
- Tensorflow: `pip install tensorflow`
  - If this doesn't work, try: `python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl`

## Usage
Run `python lander.py`
