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
