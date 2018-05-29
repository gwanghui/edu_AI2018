# Frequent Pattern Analysis
- association rule mining
- item의 빈도 분석

- Market-Basket Model
  - input
    - A Large set of items
    - A large set of baskets
  - Output
    - Rules Discovered : {Milk} --> {Coke}, {diaper, Milk} --> {Beer}
tid|items
--|--
10| Beer,Nuts,Diaper
20| Beer,Coffe,Diaper
30| Beer, Diaper, Eggs
40| Nuts, Eggs, Milk
50| Nuts, Coffee, Diaper, Eggs



- Want to discover association rules
  - People who bought {x,y,z} to {y,z}
 
## Example of Applications
-
- Items = sentences
- Baskets = a set of sentences


- Items = drugs
- Baskets = patients

- 위가 나빠서 위장약을 먹으면 위를 보호하는 위약을 먹고 간에 부담이 되니 간약을 함께 먹는다.
- 사후 추적관찰을 하면서 확인해본다.
- detect combinations of drugs

tid|items
--|--
P1| A,B,C
P2| A,B,D
P3| B,E
P4| A,C,E
P5| C,D,E

- 부작용 발견에도 많이 적용함 ex) 고산병 치료제 - 비아그라

- Finding communities in graph
  - items = outgoing nodes
  - baskets = a set of nodes
  
## Apriori Algorithm
- Itemset
  - A set of One or more Items
 

- Absolute : support or support count of X
- relative : support is the fration of transactions that contains X

- minimum_support : 우리가 결정해야 할 파라미터


- Find all the rules A -> B with minimum support and confidence
  - support s : A -> B, Support(A -> B) = P (A,B)
  - confidence c : confidence(A -> B) = P(B|A) = support_count(AUB) / support_count (A)
  
- example

tid|items
--|--
10| Beer,Nuts,Diaper
20| Beer,Coffe,Diaper
30| Beer, Diaper, Eggs
40| Nuts, Eggs, Milk
50| Nuts, Coffee, Diaper, Eggs

- Possible association rules
  - Beer -> Diaper (60%, 100%)
    - confidence : S(Beer,Diaper) = 3 / S(Beer) = 3
  - Diaper -> Beer (75%, 100%)
    - confidence : S(Beer,Diaper) = 3 / S(Diaper) = 4
  
- How Many Possible itemsets exist?
  - 1-itemset
  - 2-itemset
  - 3-itemset
  - n-itemset
  

  
