# VAMR_2024
Miniproject of course "Vision Algorithms in Mobile Robotics" by Prof. Scaramuzza, 2024

This is the implementation done by Riccardo Romani, Matteo Stürm, Alessio Vanetta

This code was tested in a conda environment running on Ubuntu 22.04. 
The full exported conda environment can be found in "environment.yml".

In order to run this code, the provided datasets "KITTI", "Malaga" and "Parking" need to be loaded into the same folder as the code in an unzipped format.

The datasets are coupled with a value in the following manner:

dataset     ds-value
KITTI       1
Malaga      2
Parking     3


To run the VO pipeline for a specific dataset select the corresponding ds-value in "main3.py" on line 21 
and then run the file "main3.py"