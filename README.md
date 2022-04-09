# SkillCorner Technical Test
This repo is the solution during the homework of SkillCorner. 
This is a kaggle challenge published by the NFL that focuses on collisions between helmets during an American football game:
[https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment)
Without trying to achieve the best performance and win the challenge, this solution aims to provide a baseline for to visualize the available data and think about/implement a first solution to the problem posed. 

# Method
These are the main steps in the algorithm:
+ Load data and process it into data frame
+ Run deepsort to track baseline-helmets video by video
+ Run homography using lines to have init matrix to map from baseline helmets to player tracking position.
+ Mapping between baseline-helmets and player tracking using findHomography + Hungarian algorithm

# Installation
- All third party modules are cloned and put to `app/modules` as submodules
## Using pip:
- Create a virtualenv:
```
python3 -m venv skillcorner
source skillcorner/bin/activate
```

- Install dependencies:
This repo require deepsort, (which requires easydict )
```
sh install.sh
pip3 install -r requirements.txt
```
## Using poetry:
- Make sure poetry is installed.
```
pip3 install poetry
```
- Install the project
```
poetry install
```

## Using docker:
- A docker file is provided for easy reproduction of the code
```
sudo docker build -t skillcorner .
```

# Usage
- Download the data and put to `input` (you will see e.g `input/nfl-health-and-safety-helmet-assignment/images`,`input/nfl-health-and-safety-helmet-assignment/test`...). If you use docker, you can mount your local dataset folder with `/skillcorner/input` folder.

## For pip:
```
{options} python3 -m app
```
 
## For poetry:
```
{options} poetry run python3 -m app 
```
## For docker:
```
sudo docker run -v {your-input-folder}:/skillcorner/input -v {your-output-folder}:/skillcorner/output skillcorner:latest
```
- There are serveral options you can set when running the code:
+ `VIDEO_ID`: default = 1: The id of test video you want to visualize. 
+ `FRAME_NUMBER`: default = 154:  The frame number of test video you want to visualize.
+ `VISUALIZE_RAW`: default = True: visualize raw data or not
+ `VISUALIZE_RESULT`: default = True: visualize result or not
- All outputs are written to 'output'. If you use docker, you can mount `/skillcorner/output` to your local output folder.


# References:
+ [Deepsort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

