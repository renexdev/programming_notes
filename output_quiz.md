####**1. Inheritance Hierarchy**
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
 
  r.print(); // Output: Inside Q.
             // There's no print() in R, so r starts looking up, and find print() in Q.
             // If, again, there's no print() in Q, then it'll call the one in P.
  return 0;
}
```
