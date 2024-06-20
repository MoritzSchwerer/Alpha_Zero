# Alpha Zero


### Goal for this project

As of now this is a working implementation of Alpha Zero with it's base
configuration as well as with the later introduced Gumbel Search.
This implementation currently uses the Pettingzoo chess environment,
which is nice to work with, but unfortunatelly seems to be the bottleneck
at least on my machine. 
My aim is to write a fast chess library in C++ and create python bindings
for it, to speed up the self play part of the algorithm significantly.
At that point I expect the DNN to be the bottleneck, which will lead to
interesting experiments in that regard.
