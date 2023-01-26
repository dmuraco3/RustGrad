# Notes


#### Survival of the fittest

can derive the functions of each neuron to determine its effect on the output

if it has a low effect on output, can remove it. 

###### this should be done after model is trained.

###### Negligable effect for small networks, useful for large networks like language transformers

if a statically sized dataset is given at run time we can check which neurons are important for the data and cut out the ones that aren't important. This could save some memory.
