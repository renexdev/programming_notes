**Print out something**:  
```js
console.log("hello world");
```

**Display a pop-up window**:  
```js
confirm("pop-up messgae!");
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
