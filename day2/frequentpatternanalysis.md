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

- 위가 나빠서 위장약을 먹으면 위를 보호하는 위약을 먹고 간에 부담이 되니 간약을 함께 먹는다.

tid|items
--|--
P1| A,B,C
P2| A,B,D
P3| B,E
P4| A,C,E
P5| C,D,E