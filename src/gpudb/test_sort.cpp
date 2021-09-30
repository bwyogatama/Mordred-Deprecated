#include <iostream>
#include <stdio.h>

#include "ssb_utils.h"

using namespace std;

int main() {
  int *h_lo_orderdate = loadColumnSort<int>("lo_orderdate", LO_LEN);
  int *h_lo_orderkey = loadColumnSort<int>("lo_orderkey", LO_LEN);

  for (int i = 59986214-1024; i < 59986214; i++) {
  	printf("%d %d\n", h_lo_orderdate[i], h_lo_orderkey[i]);
  }

  return 0;
}