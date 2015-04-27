**CSS rules**  
The following example will turn all h1 heading in html to red.  
```css
h1{
  color: red;
}
```
We can also target classes or specific elements in a class with CSS. The class can be targeted with `.` notation.   
```html
<div class="header">
  <h2>Heading</h2>
  <p>Subtitle</p>
</div>
```
```css
.header{
  color: blue;
} /* everything in header is now blue */
.header p{
  color: yellow;
} /* every <p> in header is now yellow */
```
