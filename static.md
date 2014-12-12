###**Static**

####**Static Data Member**
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
**Non-const static** data member has to be defined **outside the class** using scope resolution operator.  
Const static data member can be defined inside the class.
Also, it has to be defined **before main()**.
