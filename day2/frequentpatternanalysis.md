# Frequent Pattern Analysis
- association rule mining
- item의 빈도 분석

## Market-Basket Model
- input

tid|items
--|--
10| Beer,Nuts,Diaper
20| Beer,Coffe,Diaper
30| Beer, Diaper, Eggs
40| Nuts, Eggs, Milk
50| Nuts, Coffee, Diaper, Eggs

- Output
  - Rules Discovered : {Milk} --> {Coke}, {diaper, Milk} --> {Beer}
