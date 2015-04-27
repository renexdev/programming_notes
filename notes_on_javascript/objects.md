**Object**  
Objects in JS are pretty much the same as class objects in C++ or Java.  
We can create an object by listing it out:  
```js
//
var hunter = {
    name: "Biscuit",
    age: 57;
}
```
Or use the default object constructor:
```js
var hunter = new Object();
hunter.name = "Biscuit";
hunter.age = 57;
```
Or use a custom object constructor:  
```js
function Hunter(name,age){
    this.name = name;
    this.age = age;
}
var hunter = new Hunter("Biscuit",57);
```
We can use range-based for loop to iterate through all the properties (or property values) in an object:  
```js
for(var p in hunter){
    console.log(p);
} // "name" "age"
for(var p in hunter){
    console.log( hunter[p] );
} // "Biscuit" 57
```
`hasOwnProperty()` can be used to check if an object has a certain property.  
```js
var hunter = new Object();
hunter.name = "Biscuit";
console.log( hunter.hasOwnProperty("name") ); // true
console.log( hunter.hasOwnProperty("job") );  // false
```

**Method**  
A function associated with an object is called a method.  
```js
var hunter = new Object();
hunter.name = "Biscuit";
hunter.age = 57;
bob.setAge = function (newAge){
    hunter.age = newAge;
};
```
In JS, a method doesn't even have to be defined in an object.  
We can use **this** keyword to access stuff in the object that calls the current method.  
Here's an example:  
```js
// at this point, hunter object isn't even defined yet
var setAge = function (newAge){
    this.age = newAge;
};
// now we create hunter
var hunter = new Object();
hunter.age = 57;
// set hunter's setAge() to be the same as the one above
hunter.setAge = setAge; 
// now we can use it
hunter.setAge(18);
```

**Class prototype**  
Suppose we have a class Dog:   
```js
function Dog(breed){
    this.breed = breed;
}
// we create two objects of Dog
var Pluto  = new Dog();
var Snoopy = new Dog();
// Pluto is assigned with a method called bark()
Pluto.bark = function(){
    console.log("Woof");
}
Pluto.bark();  // now Pluto can bark...
Snoopy.bark(); // but Snoopy can't! Because only Pluto knows bark, not every Dog object
```
To mitigate this problem, use the `prototype` keyword when extending the class with new stuff:  
```js
Dog.prototype.bark = fnction(){
    console.log("Woof");
}
Snoopy.bark(); // now Snoopy can use bark()
```
