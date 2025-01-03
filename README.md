### VAMR_2024
Miniproject of course "Vision Algorithms in Mobile Robotics" by Prof. Scaramuzza, 2024

This system was implemented by [Riccardo Romani](https://github.com/riccardoromani1), [Matteo Stürm](https://github.com/mstuerm), [Alessio Vanetta](https://github.com/avanetta)


## Testing Environment
This code was tested in a conda environment running on Ubuntu 22.04. 

The system was tested on Ubuntu 22.04 using a conda environment. The hardware under which the script was run successfully consisted of a Lenovo Thinkpad X1 Yoga Gen 8 laptop with 12 Intel® Core™ i7-1355U processors of the 13th generation with 5 GHz of available Max Turbo Frequency and a Mesa Intel® Graphics (RPL-U) graphics card. 
The script ran on 150% out of 1200% (12*100%) CPU power during the development of this system.

Video recordings of the system working can be found [here](https://www.youtube.com/playlist?list=PLZFxFauWwBH2Q7nJYl93o3MuxaCO96Xxz).


## Preparation
In order to run this code, the provided datasets "KITTI", "Malaga" and "Parking" need to be loaded into the same folder as the code in an unzipped format.

The datasets are coupled with a value in the following manner:

| Dataset | ds-value |
| :---: |:---: |
| KITTI | 1 |
| Malaga | 2 |
| Parking | 3 |

The full exported conda environment can be found in "environment.yml". 
In a VSCode terminal, create the conda environment using

```bash
$ conda env create --name <envname> --file=environment.yml
```
and activate it with
```bash
$ conda activate <envname>
```


## Running the code
Make sure that you are running the code in the correct environment. Press "CTRL+SHIFT+P" and go to "Python: Select Interpreter". If not yet selected, click the previously created environment `<envname>`.

When the conda environment is active, proceed to the project folder in your console and enter

```bash
python main.py --ds <DS>
```

where `<DS>` is the corresponding value of the dataset you want to run the VO pipeline on. The script also automatically saves the plot as video in "camera_trajectory_video.avi".

You can stop the running script with keyboard input "q" in the plots or with "CTRL+C" in the console.

## Comments
During the development of this project, the full pipeline worked on the previously introduced hardware. Some additional tests were done on Windows using a different processor. In these tests, the performance of the system declined especially for the "KITTI" dataset.
