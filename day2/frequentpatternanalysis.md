# Frequent Pattern Analysis
- association rule mining
- item의 빈도 분석

## Market-Basket Model
- input
- A Large set of items
- A large set of baskets

tid|items
--|--
10| Beer,Nuts,Diaper
20| Beer,Coffe,Diaper
30| Beer, Diaper, Eggs
40| Nuts, Eggs, Milk
50| Nuts, Coffee, Diaper, Eggs

- Output
  - Rules Discovered : {Milk} --> {Coke}, {diaper, Milk} --> {Beer}

- Want to discover association rules
  - People who bought {x,y,z} to {y,z}
 
## Example of Applications

- Items = sentences
- Baskets = a set of sentences
