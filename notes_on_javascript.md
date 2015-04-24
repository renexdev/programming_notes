**Print out something**:  
```js
console.log("hello world");
```

**Display a pop-up window**:  
```js
confirm("pop-up messgae!");
```

**Prompt input from the user**:  
```js
var age = prompt("What is your age?");
```

**Variables**
```js
var a = 10;
var b = "hello";
```

**String manipulation**
```js
var email = "jasonfly07@yahoo.com";
var newEmail = email.replace("yahoo","gmail"); 
// newEmail is "jasonfly07@gmail.com"
var emailArr = email.split("@");
// emailArr is ["jasonfly07","gmail.com"]
```
```js
var str1 = "AbcDe";
var len1 = str1.length;         // len1 = 5 
var str2 = str1.substring(1,3); // str2 is "bc"
var str3 = str1.toUpperCase();  // str3 is "ABCDE"
```
**Array manipulation**
```js
var arr1 = [1,2,3];
var arr2 = [11,12,13];

// Append an item to the end of an array
arr1.push(4);      // arr1 is now [1,2,3,4]

// Append another array to the end of an array
arr1.concat(arr2); // arr2 is now [1,2,3,4,11,12,13]
```

**"True & false variables"**

| Evaluate to true | Evaluate to false |
|-------|-------|
| true  | false |
| non-zero numbers | 0 |
| "strings" | "" |
| objects | undefined |
| arrays | null |
| functions | NaN  |  
`[false]` is true, since it's an array with one element.  
`{"state":false}` is also true, since it's an object with a property called "state".  
Dont't be misled by the use of `false` keyword.

**Function**  
Functions in JavaScript looks like this:  
```js
var printName = function(name){
    console.log(name);
}
```

**Array**  
Array (or list) in JS can store different datatypes.  
```js
var arr = ["hello","world",1,2,3];  
console.log(arr[1]); // world
console.log(arr);    // ['hello','world',1,2,3]

var jagged = [[1,2,3],[true,false],"hello"]; // rows can have different length
```

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
