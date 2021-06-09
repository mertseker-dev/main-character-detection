# main-character-detection
A method for detecting the most important people in images.
## Contents
1. [Getting Started](#getting-started)
2. [Main Character Detection](#main-character-detection)
3. [Results](#results)
4. [Example Outputs](#example-outputs)
5. [References](#references)

## Getting Started
The code requires the following libraries to be installed:

-  python 3.8
-  tensorflow 2.3.1
-  opencv 4.4.0.44
-  numpy 1.18.5

The code requires OpenPose to be installed. Refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose for installation instructions. After installation, download the  script main_character_detection.py from this page. Finally, the project folder should look like:

```sh
${project_dir}/
├── openpose_models
│   └── cameraParameters
│       └── flir
│            ├── 17012332.xml.example
│   └── face
│       ├── haarcascade_frontalface_alt.xml
│       ├── pose_deploy.prototxt
│       ├── pose_iter_116000.caffemodel
│   └── hand
│       ├── pose_deploy.prototxt
│       ├── pose_iter_102000.caffemodel
│   └── pose
│       └── body_25
│           ├── pose_deploy.prototxt
│           ├── pose_iter_584000.caffemodel
│       └── coco
│           ├── pose_deploy_linevec.prototxt
│       └── mpi
│           ├── pose_deploy_linevec.prototxt
│           ├── pose_deploy_linevec_faster_4_stages.prototxt
│   ├── getModels.bat
│   ├── getModels.sh
├── outputs
├── main_character_detection.py
├── boost_filesystem-vc141-mt-gd-x64-1_69.dll
├── boost_filesystem-vc141-mt-x64-1_69.dll
├── boost_thread-vc141-mt-gd-x64-1_69.dll
├── boost_thread-vc141-mt-x64-1_69.dll
├── caffe.dll
├── caffe-d.dll
├── caffehdf5.dll
├── caffehdf5_D.dll
├── caffehdf5_hl.dll
├── caffehdf5_hl_D.dll
├── caffezlib1.dll
├── caffezlibd1.dll
├── cublas64_100.dll
├── cudart64_100.dll
├── cudnn64_7.dll
├── curand64_100.dll
├── gflags.dll
├── gflagsd.dll
├── glog.dll
├── glogd.dll
├── libgcc_s_seh-1.dll
├── libgfortran-3.dll
├── libopenblas.dll
├── libquadmath-0.dll
├── opencv_videoio_ffmpeg420_64.dll
├── opencv_world420.dll
├── opencv_world420d.dll
├── openpose.dll
├── openpose_python.py
├── pyopenpose.cp38-win_amd64.pyd
├── pyopenpose.exp
├── pyopenpose.lib
├── VCRUNTIME140.dll
```

## Main Character Detection

In order to run the code on a folder containing one or more images, run the following command:

### Usage

```sh
$ python main_character_detection.py -r=[PATH_TO_IMAGES, WRITE_PATH]
```
- PATH_TO_IMAGES (mandatory): Path to the folder that contains the images to be evaluated. The path should be in the following example format: 'C:&#92;&#92;Users&#92;&#92;user&#92;&#92;Documents&#92;&#92;examples&#92;&#92;'. Use '&#92;&#92;' in between the directories and also use '&#92;&#92;' at the end.
- WRITE_PATH (optional): Path to the folder that will contain the output images. The output images will have rectangles on the detected faces. Predicted main characters will have a green facial rectangle and the rest will have red facial rectangles. Use the same format as PATH_TO_IMAGES.

### Output format
The outputs of the evaluation will automatically be written in a csv. file with a unique timestamp name under the folder "outputs". The naming format of the files are: "YYYYMMDD-HHMMSS.csv". The output file contains the following information for each image: filename, [top left X pixel location of the main character's facial rectangle, top left Y pixel location of the main character's facial rectangle, bottom right X pixel location of the main character's facial rectangle, bottom right Y pixel location of the main character's facial rectangle]. In case there are multiple detected main characters, there will be multiple rows with the same filename where each row will contain the location of a different main character.

## Results

We collected 300 images with a wide variety from the internet. The main characters were annotated by a professional photojournalist for all of the images. Our method was able to produce a 0.83 F1-score for main character detection on this dataset.

## Example Outputs

You can see example outputs below.

![alt text](https://github.com/mertseker-dev/main-character-detection/blob/main/examples/gettyimages-1011287184-2048x2048.jpg?raw=true)

## References

- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
