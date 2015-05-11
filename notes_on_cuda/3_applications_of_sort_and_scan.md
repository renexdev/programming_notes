####**"Compact"**  
Say we have a bunch of numbers to run some process on, but we only want to process on even numbers, so we pick the even numbers out first. This is called compacting (or filtering).  
**Input**: x1, x2, x3, x4, x5  
**Predicate**: F, T, F, F, T (e.g. even or not?)  
**Sparse output**: -, x2, -, -, x5  
**Dense output**: x2, x5  
Generally dense output is preferred.  
Compact is most useful when we compact away a large number of elements, and the computation on each surviving element is expensive.

####**"Core Compact"**  
