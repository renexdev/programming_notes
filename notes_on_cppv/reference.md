###**Reference**

####**Return Values by Reference**
By making a function return by reference, we can assign value to the ouput of a function (to modify it). See the following example:
```cpp
class Image{
private:
    int** mat;
public:
    Image(int height, int width){
        mat = new int*[height];
        for(int i = 0; i < height; ++i){
            mat[i] = new int[width];
        }
    }
    int& operator()(int r, int c){
        return mat[r][c];
    }
};

int main(){

    Image im1(600, 800);

    im1(100,250) = 10;
    cout << im1(100,250) << endl;

    return 0;
}
```
By making the operator () return by reference, we can directly modify values in Image, thus avoid using set()/get() pair; this will enhance readibility. Of course, this particular example might expose data to incorrect uses, so use with caution.

####**What's the difference between pointer & reference?**
1. A pointer can be declared without pointing to anything; a reference has to bind to another object.  
2. A pointer can be reassigned; a reference cannot be reassigned.  
3. There's no "reference arithmetics".  

Rule of thumb:  
Use references in function parameters and return types.  
Use pointers in data structure and algorithm implementation.
