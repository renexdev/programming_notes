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

If we're to compact 1 to 1,000,000 with the follow 2 separate predicates:  
A. Divisible by 17 (This will remove a bunch of numbers)  
B. Not divisible by 31 (This one won't)  
Time for predicating: A = B  
Time for scanning: A = B  
**Time for scattering: A < B**  

In a more generalized form of compacting, the predicate results might be numbers rather than simple T/F. In that case, we can also use scan to produce the scatter addresses.  

####**Segmented Scan**  
Sometimes, we want to perform multiple small scans instead of one large scan. Since using one kernel for each small scan is too wasteful, what we could do instead is comcatenating all small segments into one large array, and perform a special scan call "segmented scan" on it. For example:  
**Input**: 1 2 | 3 4 5 | 6 7 8  
**Output (exclusive)**: 0 1 | 0 3 7 | 0 6 13  
Why is this useful and how is this done? The following will provide 2 examples.  

####**Sparse Matrix** 
This is how we represent a sparse matrix A (mostly with lots of zeros):    
A = [ a 0 b; c d e; 0 0 f]  
**Value**: [a b c d e f] <-(list out all non-zero values)  
**Column**: [0 2 0 1 2 1] <-(column index of all non-zero values)  
**RowPtr**: [0 2 5] <-(rows start at a, c, f, and their indices in **Value** are stored here)  

####**Spare Matrix/Dense Vector Multiplication (SpMv)**  
With sparse matrices, we can perform matrix multiplication more efficiently.  
Say we want to multiply A * B, where B is [x;y;z].  
1. Create segmented representation from **Value** and **RowPtr**.  
A_sparse = [a b | c d e | f]
2. Gather vector values using **Column**. For example, b is to be multiplied with z, which can be found out using **Column**.  
B_gathered = [x z | x y z | y]  
3. Compute pairwise multiplication (with map).  
mul = [a*x b*z | c*x d*y e*z | f*y]  
4. Finally, perform segmented scan on it!  


