all: radix-join array-join global-join
	
radix-join: radix-join.cpp
	g++ -O3 -std=c++14 -ffast-math -I../../../includes radix-join.cpp -lpthread -o radix-join

array-join:
	g++ -O3 -std=c++14 -ffast-math -I../../../includes array-join.cpp -lpthread -o array-join

global-join:
	g++ -O3 -march=native -std=c++14 -ffast-math -I../../../includes global-join.cpp -ltbb -o global-join
