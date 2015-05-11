####**"Compact"**  
Say we have a bunch of numbers to run some process on, but we only want to process on even numbers, so we pick the even numbers out first. This is called compacting (or filtering).  
**Input**: x1, x2, x3, x4, x5  
**Predicate**: F, T, F, F, T (e.g. even or not?)  
**Sparse output**: -, x2, -, -, x5  
**Dense output**: x2, x5  
Generally dense output is preferred.  
Compact is most useful when we compact away a large number of elements, and the computation on each surviving element is expensive.

####**Core Algorithms of Compacting**  
Let's assume the predicate results are: T, F, F, T, T, F, T, F  
Our task is to produce the dense output; to do this, we have to pick out all the true items and put them in the same array, which means we need to generate the **scatter addresses** of all true elements.  
The scatter addresses will then look like this: 0, -, -, 1, 2, -, 3, -  
If we convert T & F into 1 & 0, predicate would look like this: 1, 0, 0, 1, 1, 0, 1, 0  
And it turns out that compacting be done with exclusive scan on this array: 0, 1, 1, 1, 2, 3, 3, 4  
(We don't care about the results of items with false predicates)  

