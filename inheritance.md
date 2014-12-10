###**Inheritance**
Polymorphism is the provision of a **single interface** to entities of different types.  
Inheritance is one of the many ways to realize polymorphism. It's essentially about creating a derived class from a base class.

####**There are 2 kinds of inheritance.**
Virtual function & non-virtual function. See the example below:
```cpp
class Cow{
public:
	virtual void shout(){ cout << "Moo." << endl; }
};
class Werecow{
public:
	void shout(){ cout << "Aaoooohhh" << endl;}
}

int main(){
	Cow cow1;
	Werecow wcow1;
	wcow1.shout(); // This one doesn't involve virtual. Even if you take out the virtual keyword in Cow,  it would still work.
	Cow* cow2 = new Werecow;
	cow2.shout(); // This one involves virtual. 	
}
```
####**Static Type v.s. Dynamic Type**
```cpp
// D is derived from B
B* base = new D;
```
During **compile time**, base is a B pointer.  
During **runtime**, base is a D pointer.  
Static type is fixed during declaration, while dynamic type can be determined during runtime. For example,
```cpp
// someFunction() can return *D, *E, or *F
// E and F are also derived classes of B
base = someFunction();
```

####**"Pure"**
Pure virtual functions in the base class:  
1. Don't have to be implemented.  
2. Must be implemented by derived class.  
```cpp
class Shape{
public:
	// =0 is the identifier for pure
	virtual int area()=0;
}
class Square: public Shape{
private:
	int side;
public
	int area(){ return side*side; }
}
```
Beware, if a base class has a pure virtual function, this base class will become an **abstract** class and cannot be instantiated. This means that we can't pass a copy of this object to a function as input. For example,
```cpp
class Player{
public:
	virtual bool fight(Player)=0; // Compile error
}
```
To mitigate this problem, it is often recommended to pass objects as reference:
```cpp
class Player{
public:
	virtual bool fight(const Player&)=0;
}
```
####**Constructors & Destructors**
**Constructors can't be overridden** (so there's no virtual constructors). Instead, the constructor in the derived class can call the base constructor. Here's an example:
```cpp
class B{
public:
    int m_a;
    B(){
        cout << "B1" << endl;
    }
    B(int in_a): m_a(in_a){
        cout << "B2" << endl;
    }
};

class D: public B{
public:
    D(){
        cout << "D1" << endl;
    }
    D(int in_a): B(in_a){
        cout << "D2" << endl;
    }
};

int main(){
    D d1; // Output is B1 D1
    D d2(10); // Output is B2 D2
    return 0;
}
```
Destructors, on the other hand, can be overridden. This is because a derived destructor may need to clean up some member variables that are not in the base class. The derived destructor is first called, then the base destructor.

####**Object Slicing**
This will happen when you copy a derived object to a base object. The base object will lose the information from the derived class. Look at the following example.
```cpp
struct Cow{
    virtual void speak(){
        cout << "Moo." << endl;
    }
};

struct Werecow: public Cow{
    bool transformed;
    virtual void speak(){
        if(transformed) cout << "Aaooooh!" << endl;
        else cout << "Moo." << endl;
    }
};

int main(){

    Cow cow1;
    cow1.speak(); // Moo.

    Werecow wcow1;
    wcow1.transformed = true;
    wcow1.speak(); // Aaooooh!
	
    // cow2 has been "sliced", so it goes back to using Cow's shout()
    Cow cow2 = wcow1;
    cow2.speak(); // Moo.
    
    // &cow3 just points to wcow1, so cow3 can access Werecow's shout() during runtime 
    Cow &cow3 = wcow1;
    cow3.speak(); // Aaooooh!

    return 0;
}
```
