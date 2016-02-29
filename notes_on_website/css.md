### **CSS rules**  
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
.header {
  color: blue;
} /* everything in header is now blue */
.header p {
  color: yellow;
} /* every <p> in header is now yellow */
```
###**Element style**  
**Text color, font, and size**  
As shown above, we can specify the color by its name. [Here's the full list of 140 colors.](http://www.crockford.com/wrrrld/color.html)  
Alternatively, we can specify RGB or hex value.  
```css
p {
  color: rgb(130,60,0);
}
h2 {
  color: #0099cc; 
}
```
As for font and size, here's an example:  
```css
h1 {
  color: red;
  font-family: 'Shift', sans-serif;
  font-size: 48px;
}
```
**Background**  
We can customize the color or image of the background.  
```css
.jumbotron {
  background-image: url('http://imgur.com/12345');
}
```
**Border, padding & margin**  
Border is a rectangle around an element. We set the size, line style, and color of the border below.  
Padding is the space between contents and borders; increase it to enhance readability.  
Margin is the space outside border.  
All of these can be specified by either the whole thing (ex. padding) or four sides (ex. padding-top, padding-left).  
```css
.jumbotron h1 {
  padding: 23px;
  border: 3px solid #cc0000;
  margin-top: 10px;
  margin-left: 23px;
}
```
We can also use `auto` on margins to push elements to either sides or center.  
```css
.jumbotron h1 {
  margin-right: auto;
} /* this will set the header to left (maximize the right margin) */

.jumbotron h1 {
  margin-right: auto;
  margin-left: auto;
} /* this will set the header to center */
```

###**Page layout**  
Besides controlling how elements look, CSS can also control where an element sits on a page.  
  
**Display mode**  
CSS treats HTML elements as boxes. A box can be "block" or "inline".  
```css
.nav li {
  display: inline;
} /* this will set the list to be a single line */
```
**Position**  
By setting the position to be "relative", we can use "top", "left", etc., to position the elements.  
```css
.jumbotron h1 {
  position: relative;
  top: 91px;
  left: 55px;
}
```
We can also use float to pull the elements to either far left or far right.  
```css
img {
  float: right;
}
```
