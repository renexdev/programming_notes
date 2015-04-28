**Five basic elements in a website**  
*1. Heading*  
There're 6 levels, `h1` to `h6`.  
```html
<h1>Bible</h1>
```
*2. Paragraph*  
```html
<p>In the beginning God created...</p>
```
*3. Links*  
First the link, then the text.  
```html
<a href="https://www.bible.com/genesis"> Click to read</a>
```
*4. Images*   
First the attribute, then the link to the image.
```html
<img src="http://goo.gl/abc123">
```
*5. Lists*  
Use `<ul>` to describe a bulleted list. Every item in the list comes with `<li>`.  
```html
<ul>
  <li>Genesis</li>
  <li>Exodus</li>
</ul>
```

**Structure of a web page**  
Everything inside a website is inside `<html>`.  
`doctype` tells the browser the version of HTML. It's not part of HTML, so no need to close it.  
You can specify the CSS file in `<head>`.  
The contents of the web page are in `<body>`.  
```html
<!DOCTYPE html>
<html>
  <head>
    <link href="main.css" rel="stylesheet"/>
  </head>
  <body>
    <h1>Bible</h1>
    <p>In the beginning...</p>
  </body>
</html>
```
Use `<div>` to group & organize the elements:  
```html
<div class="container">
  <ul>
    <li>Sign Up</li>
    <li>Log In</li>
    <li>Help</li>
  </ul>
</div>
```
