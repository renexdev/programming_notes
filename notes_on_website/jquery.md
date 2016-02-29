**jQuery** is a javascript library for various HTML interactions, such as event handling, animation, etc.  

### An example 
```js
$(document).ready(function() {
  $('div').hide();
});
```
`$(document)` is a jQuery object. We use `$()` to create a jQuery object.  
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

### Different ways for selection  
**Directly selecting an HTML element**  
This is shown above.  
  
**Selecting by class**  
```html
<body>
  <div class="block1"></div>
  <div class="block2"></div>  
</body>
```
```js
$(document).ready(function() {
  $('.block1').fadeOut('slow');
});
```
For a class, put a `.` in front of class name.  
You can even do a compound selection like this:  
```js
$(document).ready(function() {
  $('.block1, .block2').fadeOut('slow');
});
```


**Selecting by id**  
```html
<body>
  <div id="blue"></div>
</body>
```
```js
$(document).ready(function() {
  $('#blue').fadeOut('slow');
});
```
For an id, put a `#` in front of class name.  
