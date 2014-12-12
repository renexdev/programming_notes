###**Static**
This keyword has 3 different usages in C++:   
1. Used inside a *function*.  
2. Used inside a *class*.  
3. Used inside a file as a *global variable*.  
In any case, static variable always has to be defined **before main()**.

####**1. Uses inside a function**
```cpp
void func(){
    for(int i=0; i<3; i++){
        static int x = 0; // This line will run only once.
        x++;
        cout << x << endl;
    }
}

int main(){

    func();
    func();
    func();

    return 0;
}
```
Output: 1 2 3 4 5 6 7 8 9.  
Beware, although x will be inside the memory until the programs is killed, its scope is the same as an ordinary local variable.  
Why do we need this? For example, your function may need to memorize the maximum returned value thus far. You can use a static local variable to store it.

####**2. Used inside a Class**
```cpp
class Point {
private:
    int x;
    int y;
    static int count;
    static const int max = 100;
public:
    Point(int i=0, int j=0): x(i), y(j){
        count++;        
    }
    static void printCount(){
        cout << count << endl;
    }
    static void printMax(){
        cout << max << endl;
    }
};

int Point::count = 0;

 
int main(){
   Point p1(1,2);
   Point p2(4,7);
   Point p3(0,-1);

   Point::printCount();
   Point::printMax();

   return 0;
}
```
Output: 3 100
**Non-const static** data member has to be defined **outside the class** using scope resolution operator.  
Const static data member can be defined inside the class.  
Why do we need this? For example, your class may need to keep track of how many instances of this class there are; a static counter variable is useful in this case.

####**3. Used inside a File as a Global Variable**
We know there's **local scope** and **global scope**; if we put static in front of a global variable, we can create something in between: **file scope**. This means that this variable can only be accessed in this file.
