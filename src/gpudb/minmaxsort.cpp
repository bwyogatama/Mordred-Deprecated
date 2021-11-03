#include "ssb_utils.h"
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

using namespace std;

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "col-name len" << endl;
    return 1;
  }

  string col_name = argv[1];
  int len = atoi(argv[2]);

  cout << atoi(argv[2]) << endl;

  uint *raw = loadColumnSort<uint>(col_name, len);
  // if (len == LO_LEN && SF == 20) {
  //   for (int j = 104857600; j < 104858624; j++) {
  //     raw[j] = raw[104857599];
  //   }
  // }
  cout << "Loaded Column " << col_name << endl;

  ofstream myfile;
  myfile.open (DATA_DIR + col_name + "minmax");

  int total_segment = ((len + SEGMENT_SIZE - 1)/SEGMENT_SIZE);

  cout << len << endl;

  for (int i = 0; i < total_segment; i++) {
  	int adjusted_len = SEGMENT_SIZE;
  	if (i == total_segment-1) {
  		adjusted_len = len - SEGMENT_SIZE * i;
  	}

    int min = raw[i*SEGMENT_SIZE];
    int max = raw[i*SEGMENT_SIZE];
  	for (int j = 0; j < adjusted_len; j++) {
  		if (raw[i*SEGMENT_SIZE + j] > max) max = raw[i*SEGMENT_SIZE + j];
  		if (raw[i*SEGMENT_SIZE + j] < min) min = raw[i*SEGMENT_SIZE + j];
  	}
  	myfile << min << " " << max << '\n';
  }

  myfile.close();

  return 0;
}