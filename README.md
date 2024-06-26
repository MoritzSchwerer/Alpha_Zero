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


### Notes for me:

AlphaZero uses a batch size of 4096 and it is trained for 700k step,
meaning it sees ~2.8B samples. At 44M games that means if each game
has the same weight, it will be seen roughly 65 times by the network.
This would equate to having a replay_buffer of 1M games, training on
3M samples out of the 1M games, then replacing the oldest 50K games
by new self play games and repeat...

With the random network in the beginning we want to run 300 - 500 steps
per game before terminating.
Later on the games might not last that many steps so we can probably cut
that number down over time.
