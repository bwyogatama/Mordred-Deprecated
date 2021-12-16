#include "ssb_utils.h"
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

using namespace std;

int main(int argc, char** argv) {
  if (argc != 4) {
    cout << "col-name len SF" << endl;
    return 1;
  }

  string col_name = argv[1];
  int len = atoi(argv[2]);
  string sf = argv[3];

  cout << atoi(argv[2]) << endl;

  uint *raw = loadColumn<uint>(col_name, len);
  // if (len == LO_LEN && SF == 20) {
  //   for (int j = 104857600; j < 104858624; j++) {
  //     raw[j] = raw[104857599];
  //   }
  // }
  cout << "Loaded Column " << col_name << endl;

  ofstream myfile;
  myfile.open ("/home/ubuntu/Implementation-GPUDB/test/ssb/data/s" + sf + "_columnar/" + col_name + "minmax");

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

// void tokenize(string s, string del = " ")
// {
//     int start = 0;
//     int end = s.find(del);
//     while (end != -1) {
//         string minstring = s.substr(start, end - start);
//         cout << stoi(minstring) << endl;
//         start = end + del.size();
//         end = s.find(del, start);
//     }
//     string maxstring = s.substr(start, end - start);
//     cout << stoi(maxstring) << endl;
// }

// int main(int argc, char** argv) {

//   if (argc != 2) {
//     cout << "col-name" << endl;
//     return 1;
//   }

//   string col_name = argv[1];

//   string line;
//   ifstream myfile (DATA_DIR + col_name + "minmax");
//   if (myfile.is_open())
//   {
//     while ( getline (myfile,line) )
//     {
//       cout << line << '\n';
//       tokenize(line);
//     }
//     myfile.close();
//   } else {
//     cout << "Unable to open file"; 
//     assert(0);
//   }

//   return 0;
// }