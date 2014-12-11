####**1. Multi-level Inheritance Hierarchy**
```cpp
using namespace std;
class P {
public:
   void print()
   { cout <<" Inside P::"; }
};
 
class Q : public P {
public:
   void print()
   { cout <<" Inside Q"; }
};
 
class R: public Q {
};
 
int main(void){
  R r;
 
  r.print();
  return 0;
}
```
Output: Inside Q.  
There's no print() in R, so r starts looking up, and find print() in Q.  
If, again, there's no print() in Q, then it'll call the one in P.

2. 
```cpp
class Base{
public:
  Base()
  {
    fun();
  }
  virtual void fun()
  {
    cout<<"Base Function";
  }
};
 
class Derived: public Base{
public:
  Derived(){}
  virtual void fun()
  {
    cout<<"Derived Function";
  }
};
 
int main(){
  Base* pBase = new Derived();
  delete pBase;
  return 0;
}
```
Output: Base Function.  
If:  
1. The virtual function is called inside the constructor or destructor  
2. The object the under construction  
then the function called will be the one in its own class, not the one overriding it.  
It is generally recommended that don't call a virtual function inside constructor or destructor.
