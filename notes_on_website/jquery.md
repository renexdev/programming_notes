**jQuery** is a javascript library for various HTML interactions, such as event handling, animation, etc.  

### An example 
```js
$(document).ready(function() {
  $('div').hide();
});
```
`$(document)` is a jQuery object.  
As soon as it's ready, it'll perform the action specified by `function()`.  
```js
$(document).ready(function() {
  var $target = $('div');
  $target.click(function() {
    $target.fadeOut('slow');
  });
});
```
Here we use the variable `$target` for `$('div')`.  
The use of `$` in front of the variable name is just a convention is signify that this variable contains a jQuery object.  
Also, a function can take another function as its argument, like `click()` above.  

