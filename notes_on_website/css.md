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
**Text color, font, and size**  
As shown above, we can specify the color by its name. [Here's the full list of 140 colors.](http://www.crockford.com/wrrrld/color.html)  
Alternatively, we can specify RGB or hex value.  
```css
p{
  color: rgb(130,60,0);
}
h2{
  color: #0099cc; 
}
```
As for font and size, here's an example:  
```css
h1{
  color: red;
  font-family: 'Shift', sans-serif;
  font-size: 48px;
}
```
**Background**  
We can customize the color or image of the background.  
```css
.jumbotron{
  background-image: url('http://imgur.com/12345');
}
```
**Border**
```css
.jumbotron h1 {
  border: 3px solid #cc0000;
}
```
The example above set the size, line style, and color of the border around the header.  
