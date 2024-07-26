# Alpha Zero


### Goal for this project

As of now this is a working implementation of Alpha Zero with it's base
configuration as well as with the later introduced Gumbel Search.
This implementation currently uses the Pettingzoo chess environment,
which is nice to work with, but unfortunatelly seems to be the bottleneck
at least on my machine. 


This repo implements Gumbel Search from this [paper](https://openreview.net/forum?id=bERaNdoegnO).
The code can be read in `gumbel_alpha_zero.py` but unfortunatelly to make the code 
as fast as possible I had to sacrifice readability. So I would not recommend trying to
read the code there, since it implements a batched version of the algorithm so that
the gpu would be used more effectively.


I developed my own C++ Chess Environment which is supposed to be a drop in
replacement for [Pettingzoo](https://pettingzoo.farama.org/environments/classic/chess/),
which will allow me to train a lot more effectively and hopefully reach
a high level of performance with limited resources. As of today I have not
incorporated this my own Chess Environment but it is on top of my list.

At that point I expect the DNN to be the bottleneck, which will lead to
interesting experiments in that regard.

