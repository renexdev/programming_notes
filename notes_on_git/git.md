###**Notes on Git**

####**Compare 2 commits**
```cpp
git diff <commit id 1> <commit id 2>
```
It's  similar to the "diff" tool:
```cpp
diff -u <file 1> <file 2>
```
But git diff is for comparing 2 versions of the same file.

####Keeps commits short, but not one-line-change short.
A good rule of thumb is to make one commit per logical change. For example, if you fixed a typo, then fixed a bug in a separate part of the file, you should use one commit for each change since they are logically separate.
