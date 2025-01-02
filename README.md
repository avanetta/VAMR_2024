# VAMR_2024
Miniproject of course "Vision Algorithms in Mobile Robotics" by Prof. Scaramuzza, 2024

This is the implementation done by Riccardo Romani, Matteo Stürm, Alessio Vanetta

This code was tested in a conda environment running on Ubuntu 22.04. 


The system was tested on Ubuntu 22.04 using a conda environment. The full exported conda environment can be found in "environment.yml". 
The hardware under which the script was run successfully consisted of a Lenovo Thinkpad X1 Yoga Gen 8 laptop with 12 Intel® Core™ i7-1355U processors of the 13th generation and a Mesa Intel® Graphics (RPL-U) graphics card.



In order to run this code, the provided datasets "KITTI", "Malaga" and "Parking" need to be loaded into the same folder as the code in an unzipped format.

The datasets are coupled with a value in the following manner:

dataset     ds-value
KITTI       1
Malaga      2
Parking     3


You can create the conda environment using 

conda env create --name <envname> --file=environment.yml

and activate it with

conda activate <envname>


When the conda environment is active, proceed to the project folder in your console and enter

python main3.py --ds <DS>

where <DS> is the corresponding value of the dataset you want to run the VO pipeline on.

You can stop the running script with keyboard input "q" in the plots or with "CTRL+C" in the console.
