**Print out something**:  
```js
console.log("hello world");
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
