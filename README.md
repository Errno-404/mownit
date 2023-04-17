# Numerical methods

## Navigation

- [About this repo](#about)
- [Basic interpolation](#lab2)
- [Hermite interpolation](#lab3)
- [Cubic and quadratic spline interpolation](#lab4--hard-)
- [How to install & run](#running-examples)

## About

Repository contains some numerical methods such as interpolation, approximation, fft etc.
The following `lab` directories contain all the code written by myself for the Numerical Methods classes at AGH UST.
In the future I plan to write more about those methods.

## Note

_There is a lot of duplications of the code due to the fact that each lab has to be a completely new project.
When this term ends, I'll probably clean this code up, make it more generic and remove duplications._

## lab2

Contains an example of an interpolation using both Newton and Lagrange methods (polynomials).

## lab3

Contains an example of Hermite interpolation.

## lab4 (hard)

Contains examples of spline interpolation using two methods:

- cubic interpolation (with natural and clamped boundary conditions)
- quadratic interpolation (with natural and clamped boundary conditions)

## Running examples

To be able to run and test those examples you'll need to follow these instructions:

1. Make sure You have the latest Python interpreter installed on your machine. You can check if it is installed by
   running `terminal` if You are on Linux or `cmd` if you are on Windows. Type `python` or `python3` to check if it is
   installed.
2. To be able to run those examples You will also need to have `numpy` and `matplotlib` libraries installed. You can
   install them by typing `pip install numpy` and `pip install matplotlib` in your system console (**not in the Python
   interpreter!**)
3. If the gods are on your side, all You have to do now is to run the script You are interested in. To do so, just open
   your `terminal` or `cmd` and type `python <name_of_the_script>` or `python3 <name_of_the_script>`. Remember to
   add `.py` to the name of the script!
4. For example if You want to run _hermite_, You have to run `python hermite.py`.
