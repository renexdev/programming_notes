###**Constructors( & Destructors)**

####**Does compiler create default constructor when we define our own?**
No. So always remember to include a default one.

####**Copy Constructor**
There's a default copy constructor in your class.
You don't need to explicitly create a copy constructor **unless** your class contains member variables that are pointers, and you want to achieve **deep copy**.  
Here's an example: the output will be different if the copy constructor is removed.
```cpp
class Hero{
public:
    int* m_HP;

    Hero(int* iPtr = NULL){
        m_HP = iPtr;
    }
    Hero(const Hero& anotherHero){
        *m_HP = *(anotherHero.m_HP);
    }
    void print(){ cout << *m_HP << endl; }
    void setHP(int* x){ m_HP = x; }
};

int main(){
    int x = 5;
    Hero hero1 = Hero(&x);
    Hero hero2 = hero1;

    hero1.print();
    hero2.print();
    // Output will be 5 5

    x = 10;
    hero1.setA(&x);
    
    hero1.print();
    hero2.print();
    // If we keep the copy constructor, output will be 10 5 (deep copy)
    // If we remove the copy constructor, output will be 5 5
    // This is because if we don't specifically make hero2 copy the content of *m_HP with
    // our own copy constructor, the default copy constructor will merely copy the pointer itself,
    // thus create a shallow copy of hero1.

    return 0;
}
```
Beware, the argument passing to copy constructor has to be const reference.

####**Copy Constructor v.s. Assignment Operator**
They both achieve similar effects, but are different stuff. See below:
```cpp
class Hero{
private:
    int HP;
public:
    Hero(){}
    Hero(int in_HP): HP(in_HP){}
    void printHP(){ cout << HP << endl; }
};

int main(){
    Hero h1(100);
    Hero h2 = h1; // This calls (default) copy constructor
    Hero h3;
    h3 = h1; // This calls (default) assignment operator

    h2.printHP();
    h3.printHP();

    return 0;
}
```
####**When do we need to write a user-defined destructor?**
When our class has dynamically allocated memory or pointer. We'd want to release the memory in our destructor.

####**Conversion Constructor**
If a class has a constructor that takes a single argument, then we can use it as a "conversion constructor". It's just some syntax sugar.
```cpp
class Girl{
private:
    int age;
public:
    Girl(){}
    Girl(int in_age): age(in_age){}
    void printAge(){ cout << age << endl; }
};

int main(){
    Girl Cindy(17); // This is the usual way
    Girl Kate = 18; // Here we use is as a conversion constructor
    Kate.printAge(); // Output: 18
    
    return 0;
}
```
