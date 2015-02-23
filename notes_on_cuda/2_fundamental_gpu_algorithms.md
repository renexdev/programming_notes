####**Steps & Work**  
When we talk abour algorithms, we deal with 2 costs: **step complexity** & **work complexity**.  
A parallel program is considered *efficient* (compared with serial implementation) if step complexity is reduced, while the overall work complexity is relatively the same.  

####**Reduce**  
Reduce has 2 inputs: **sets of elements** and a **reduction operator**.  
For example, Reduce[(1,2,3,4),'+']=10.  
The operator has to be both **binary** & **associative**. (So `a+b`, `a||b`, `min(a,b)` are ok, but `pow(a,b)`, `a/b` are not.)  

If we are summing up n elements, the serial reduce looks like this:  
```cpp
int sum = 0;
for(int i=0; i<a.size(); i++){
  sum += a[i];
}
```
Pretty straightforward.  
Both the number of steps and total amount of work are n-1, and the complexity is O(n).  

