A new function `CalculateMilkProduct()` takes input `Cow` and return the amount of milk that can be generated.  
This requires a lot of `Cow.ComputeMilk()` calls in `CalculateMilkProduct()`, so we decide to give `Cow` a private
data member `m_Milk`, and call `ComputeMilk()` everytime a `Cow` object is created. This way, we can simply use a
getter to access `m_Milk` in `Cow`.  

But is this a good thing?  

It turns out that a lot of other functions and classes that use `Cow` will be slowed down by this. For example,
a function called `CountCow()` doesn't need to know how much milk a `Cow` has, so computing it isn't necessary.  
Depending on the situation, it's viable to create a new class `MilkCow` and design your `CalculateMilkProduct()` around it.
