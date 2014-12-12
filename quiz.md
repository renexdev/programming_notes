####**1. Scope Resolution Operator**
```cpp
class A {
public:
    static int i;
};
int A::i = 10;
 
namespace B{
    int i = 20;
}

int i = 30;

int main(){
    cout << A::i << endl;
    cout << B::i << endl;
    cout <<  ::i << endl; 

    return 0;
}
```
Output: 10 20 30.  
Beware that ::i will simply call the global variable.

####**2. Default Constructor?**
```cpp
class Test{
public:
  Test(){ cout << "Called default constructor." };
};

int main(){
    Test t1(); // () but no argument?
    return 0;
}
```
Output: <nothing>.
Test t1() is not calling the default constructor of Test at all.  
Compiler will interpret this as declaring a function "t1()" that returns a Test object.
