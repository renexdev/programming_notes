###**Notes on Git**

####**Compare 2 commits**
```bash
git diff <commit ID 1> <commit ID 2>
```
It's  similar to the "diff" tool:
```bash
diff -u <file 1> <file 2>
```
But git diff is for comparing 2 versions of the same file.  
  
To get color output,
```bash
git config --global color.ui auto
```

####**Keeps commits short, but not one-line-change short.**
A good rule of thumb is to make *one commit per logical change*.  
For example, if you fixed a typo, then fixed a bug in a separate part of the file, you should use one commit for each change since they are logically separate.

####**Checkout a previous commit**
```bash
git checkout <commit ID>
```
We can use this for debugging.

####**Check the status of git**  
```bash
git status
```
It tells us which branch we're on, what's changed since the last commit, and a list of untracked files.  

####**Choosing what to commit**  
*Working directory* → *Staging area* → *Repository*  
Use `git add` to move stuff into staging area, then use `git commit` to move them into repository.  










