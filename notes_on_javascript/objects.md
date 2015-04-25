**Object**  
Objects in JS are like classes.  
We can create an object by listing it out:  
```js
//
var hunter = {
    name: "Biscuit",
    age: 57;
}
```
Or use an object constructor:
```js
var hunter = new Object();
hunter.name = "Biscuit";
hunter.age = 57;
```
We can use range-based for loop to iterate through all the stuff in an object:  
```js
for(var i in hunter){
    console.log(i);
} // 'Biscuit' 57
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
