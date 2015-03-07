####**Steps & Work**  
When we talk abour algorithms, we deal with 2 costs: **step complexity** & **work complexity**.  
A parallel program is considered *efficient* (compared with serial implementation) if step complexity is reduced, while the overall work complexity is relatively the same.  

####**Reduce**  
Reduce has 2 inputs: **sets of elements** and a **reduction operator**.  
For example, Reduce[(1,2,3,4),'+']=10.  
The operator has to be both **binary** & **associative**. (So `a+b`, `a||b`, `min(a,b)` are ok, but `pow(a,b)`, `a/b` are not.)  

**Serial reduce** is essentially ((a+b)+c)+d. Step complexity is O(n).  
**Parallel reduce**, on the other hand, is (a+b)+(c+d), which means complexity is O(log(n)). (Summing up 1024 elements only requires 10 steps.)  

See `reduce.cu` for the code snippets of reducing with globabl or shared memory.  
In the above example, reducing with global memory uses 3 times more memory than the shared version.  


