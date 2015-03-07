####**Steps & Work**  
When we talk abour algorithms, we deal with 2 costs: **step complexity** & **work complexity**.  
A parallel program is considered *efficient* (compared with serial implementation) if step complexity is reduced, while the overall work complexity is relatively the same.  

####**Reduce**  
Any operation(reduce, scan, etc.) has 2 inputs: **an input array** and an **operator**.  
For example, Reduce[(1,2,3,4),'+']=10. (1,2,3,4) is the array, and '+' is the operator.   
The operator has to be both **binary** & **associative**. (So `a+b`, `a||b`, `min(a,b)` are ok, but `pow(a,b)`, `a/b` are not.)  

**Serial reduce** is essentially ((a+b)+c)+d. Step complexity is O(n).  
**Parallel reduce**, on the other hand, is (a+b)+(c+d), which means complexity is O(log(n)). (Summing up 1024 elements only requires 10 steps.)  

See `reduce.cu` for the code snippets of reducing with globabl or shared memory.  
In the above example, reducing with global memory uses 3 times more memory than the shared version.  

####**Scan**  
Here's an example of scan: Scan[(1,2,3,4),'+']=(1,3,6,10).(cumulative sum)  
Though not obvious, it's very useful in parallel.  
There're 2 kinds of scan: *exclusive* & *inclusive*.  
Exclusive: (0,1,3,6). (The nth output excludes the nth input.)  
Inclusive: (1,3,6,10).  
**Serial scan** is, again, an O(n) operation, both in work and step.  
**Paralell scan** is a bit more complicated.  

####**Parallel Scan**  
One way to perform scan(on an N-element array) is to run N reduce at the same time. For example, we run the Kth reduce from the 0th element to the (K-1)th element. This way:  
Step: O(log(n)), since when analyzing algorithm complexity, we assume infinite GPU resources.  
Work: O(n^2). Since the Kth reduce requires K operations, the sum of all operations is roughly 0.5*K^2.  

