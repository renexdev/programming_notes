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
Output: (nothing).  
Test t1() is not calling the default constructor of Test at all.  
Compiler will interpret this as declaring a function "t1()" that returns a Test object.

####**3. "Uses Undefined Class"**
```cpp
class A;

class B{
public:
  void foo(A& ar);
private:
  A a;
};

class A{
//something
};
```
Result: compile error C2079.  
Since compiler doens't know how big A is until it sees the full definition of A, we can't just create an A object in B like this. We can either move the definition of A ahead of B, use a pointer to A, or directly define A in B.  
Note that a reference of A in B is fine, so foo() won't generate any error.
