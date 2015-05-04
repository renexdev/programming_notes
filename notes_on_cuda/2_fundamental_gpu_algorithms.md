####**0. Steps & Work**  
When we talk abour algorithms, we deal with 2 costs: **step complexity** & **work complexity**.  
A parallel program is considered *efficient* (compared with serial implementation) if step complexity is reduced, while the overall work complexity is relatively the same.  
In this section, we would go over three fundamental GPU algorithms: reduce, scan, and histogram.  

####**1. Reduce**  
Any operation(reduce, scan, etc.) has 2 inputs: **an input array** and an **operator**.  
For example, Reduce[(1,2,3,4),'+']=10. (1,2,3,4) is the array, and '+' is the operator.   
The operator has to be both **binary** & **associative**. (So `a+b`, `a||b`, `min(a,b)` are ok, but `pow(a,b)`, `a/b` are not.)  

**Serial reduce** is essentially ((a+b)+c)+d. Step complexity is O(n).  
**Parallel reduce**, on the other hand, is (a+b)+(c+d), which means complexity is O(log(n)). (Summing up 1024 elements only requires 10 steps.)  

See `reduce.cu` for the code snippets of reducing with globabl or shared memory.  
In the above example, reducing with global memory uses 3 times more memory than the shared version.  

####**2. Scan**  
Here's an example of scan: Scan[(1,2,3,4),'+']=(1,3,6,10).(cumulative sum)  
Though not obvious, it's very useful in parallel.  
There're 2 kinds of scan: *exclusive* & *inclusive*.  
Exclusive: (0,1,3,6). (The nth output excludes the nth input.)  
Inclusive: (1,3,6,10).  
**Serial scan** is, again, an O(n) operation, both in work and step.  
**Paralell scan** is a bit more complicated.  

####**Parallel Scan**  
One way to perform scan(on an N-element array) is to run N reduce at the same time. For example, we run the Kth reduce from the 0th element to the (K-1)th element. This way:  
Step: log(n), since when analyzing algorithm complexity, we assume infinite GPU resources.  
Work: O(n^2). Since the Kth reduce requires K operations, the sum of all operations is roughly 0.5*K^2.  
This is ridiculously inefficient, so we need some other approaches.  

####**Hillis Steele Inclusive Scan**  
**Starting with step 0. On step K, add yourself to your 2^K right neighbor.**  
The following is an example:  

|Original|1|2|3|4|5|6|7|8|
|---|---|---|---|---|---|---|---|---|
|After step 0|1|3|5|7|9|11|13|15|
|After step 1|1|3|6|10|14|18|22|26|
|After step 2|1|3|6|10|15|21|28|36|  

Step: log(n).  
Work: O(nlog(n)). (This is not good!)  

####**Blelloch Exclusive Scan**  
See [this paper](http://www.cs.cmu.edu/~./blelloch/papers/Ble93.pdf) for a detailed explanation.  

Step: 2log(n).  
Work: O(n). (This is good!)  

####**Which Scan Should I Use?**  
If there're **more work than processors**, pick the work effcient scan. (**Blelloch**)  
If there're **more processors than work**, pick the step efficient scan. (**H/S**)  
BTW, if there's only one processor, you have to use serial scan.  

####**3. Histogram**  
Histogram is distributing n items into b bins. A naive parallel approach is having n threads running at the same time.  
This obviously will not work because of data race. (2 threads trying to increment a bin with 5 will result in a pair of 6 being poured into it, not 7)  

####**Histogram with Atomic Operations**  
Naurally, one way to solve this is using atomic operations to ensure that only one thread is reading/writing a bin at a time. 
```cpp
__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT){
  int myID = threadIdx.x + blockDim.x * blockIdx.x;
  int myItem = d_in[myId];
  int myBin = myItem % BIN_COUNT;
  atomicAdd(&(d_bins[myBin]), 1);
}
```
The drawback of this approach this that atomic operations serialize the read/write operations a lot, so the fewer bins there are, the less efficient it is. Imagine if there are 1000 bins vs 100 bins. In the former cases, there can be at most 1000 threads accessing bins at the same time, while only 100 threads in the latter case.  
In general, an algorithm relying on atomic operations will limit its parallelism, and thus its scalability. No matter how many threads your GPU is capable of running at the same time, you're limited to, in this case, the number of bins.  

####**Local Histogram Then Reduce**  
Here's a more efficient approach: if there're 128 items, 8 threads and 3 bins, we can divide the items into 8 groups, and let each thread take care of 16 items. Each item will maintain its own "local" 3-bin histogram. Once all threads are done with their binning, we reduce all 8 local histograms.  
There's no need for atomic operations, since each thread computes its local histogram serially.  

####**Sort Then Reduce By Key**  
Finally, we can first sort each item by its bin number, then perform reduce one bin number at a time. More on this later.  
