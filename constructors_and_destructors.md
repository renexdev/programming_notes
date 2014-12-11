###**Constructors & Destructors**

####**Copy Constructor**
There's a default copy constructor in your class.
You don't need to to explicitly create one **unless** your class contains member variables that are pointers, and you want to
achieve **deep copy**.  
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
    // If we keep the copy constructor, output will be 10 5
    // If we remove the copy constructor, output will be 5 5, which is not desirable

    return 0;
}
```
