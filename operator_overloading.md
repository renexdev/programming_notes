###**Operator Overloading**

####**When should we write our own assignment operator?**
The same as copy constructor: when we need to deal with pointers in our class.

####**Conversion Operator**
If we want to convert our object to a primitive type or another class, we can use conversion operator. See the example below.
```cpp
class Fraction{
private:
    int numer;
    int denom;
public:
    Fraction(int n, int d): numer(n), denom(d){
        cout << "Created: ";
        cout << numer << "/" << denom << endl;
    }
    // Here we create a conversion operator to convert out Fraction to double
    operator double(){
        return static_cast<double>(numer)/denom;
    }
};

int main(){
    Fraction Frac(2, 5);

    double d = Frac;
    cout << d << endl; // Output: 0.4
    
    return 0;
}
```
