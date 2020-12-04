# NASRL
## Requirements
This project requires `python 3.8`. 
It may work on 3.9, but has been explicitly marked as incompatible with `python <= 3.8`. 
If this is an issue, we can distribute a Docker container or VM containing a fully-configured version of our project.

The project has been tested on Ubuntu 20.04.01 LTS and Mac OS 11.0.1 [See 1]. We can offer no insight as to whether to project will run on Windows as is.

## Installation 
Create a new virtual environment for python 3.8 and activate it.
Collect our dependencies using  `pip install -r requirements.txt`.
Build & install our library using `pip install -e .` from the root of the project.
The bulk of the RL code comes from on of our subprojects [`librl`](https://github.com/Matthew-McRaven/librl) , which is installed via Pip.

## Testing
You can run various configurations via `tools/run.py`.
This script requires a directory in which to log runtime statistics.

From you you can choose which RL algorithm you would like to use:
* VPG
* PGB
* PPO

As well as what form of networks:
* FC only
* CNN only
* CNN+FC joint

There appear to be some `NaN` issues with PGB/PPO and CNN/CNN+FC.
I have narrowed it down to incorrect log_prob computations, but did not have time to fix them.
However, VPG should work for any form of network, and all algorithms work on FC only.


## Notes
[1] Installing torch, numpy, scipy on Mac OS 11 can be a huge pain due to PyPi not having wheels for macosx_11_x86_64. 
If this affects you, contact us.
