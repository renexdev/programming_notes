###**Chapter 4: Multi-File Programs, Abstraction, and the Preprocessor**  

* Linking error usually occurs when a function is prototyped but not implemented. It has nothing to do with syntax error.  
* **Abstraction** means the simplification of an object that allows it to be used without an understanding of its underlying logic. The complexity of the implementation is hidden behind a very simple interface.  
* The **preprocessor** will modify your codes before compiling. `#include` is one of the most common preprocessing commands.
* Preprocessor was developed in the early days of C, and a lot of its functionalities are deprecated by new features of C/C++, so minimize the use of preprocessor.  
* A common way to block out codes is to use `#if 0`...`#endif`.  
* Use **include guards** to prevent doubly-defining something:
```cpp
#ifndef Foo_Included
#define Foo_Included
//...
#endif
```
* In general, don't use **macros**. If you want to inline a function, just use the keyword `inline`.  
* Special preprocessor values: `__TIME__`, `__DATE__`, `__FILE__`(current file name), `__LINE__`(current line number), etc.


