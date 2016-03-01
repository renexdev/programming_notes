### Modifyint HTML structure  
**Insert element inside something**
```js
$(document).ready(function () {
  $("body").append("<p>foo</p>");
});
```
**Insert element before/after something**
```js
$(document).ready(function () {
  $('#one').before('<p>foo1</p>');
  $('#one').after('<p>foo2</p>');
});
```
**Move element around**  
Just use `after()` or `before()` on existing elements.  
```js
$(document).ready(function () {
  // js is pass by reference (or more specifically, pass a copy of reference)
  var $paragraph = $("p"); 
  $('#two').after($paragraph);
});
```
**Removing & emptying elements**  
```js
$(document).ready(function () {
  $('.container').empty();  // this will clear the contents inside it
  $('.container').remove(); // this will delete the element completely
});
```
