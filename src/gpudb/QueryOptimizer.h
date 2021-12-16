#ifndef _QUERY_OPTIMIZER_H_
#define _QUERY_OPTIMIZER_H_

#include "CacheManager.h"
#include "KernelArgs.h"
#include "common.h"

#define NUM_QUERIES 13
#define MAX_GROUPS 128

class CPUGPUProcessing;

enum OperatorType {
    Filter, Probe, Build, GroupBy, Aggr, CPUtoGPU, GPUtoCPU, Materialize, Merge
};

enum DeviceType {
    CPU, GPU
};

class Zipfian {
public:

	int seed;
	double c;
	double alpha;
	long x;
	int n;
	int range;
	const long  a =      16807;  // Multiplier
	const long  m = 2147483647;  // Modulus
	const long  q =     127773;  // m div a
	const long  r =       2836;  // m mod a

	pair<int, int> year;
	pair<int, int> yearmonth;
	pair<int, int> date;

	Zipfian (int N, int Range) {
		seed = 123;
		alpha = 2.0;
		x = seed;
		n = N;
		range = Range;

	  	c = 0;
		for (int i=1; i<=N; i++) c = c + (1.0 / pow((double) i, alpha));
		c = 1.0 / c;
	};


	int zipf()
	{
		// static int first = TRUE;      // Static first time flag
		// static double c = 0;          // Normalization constant
		double z;                     // Uniform random number (0 < z < 1)
		double sum_prob;              // Sum of probabilities
		double zipf_value;            // Computed exponential value to be returned

		// Pull a uniform random number (0 < z < 1)
		do {
			z = rand_val();
		} while ((z == 0) || (z == 1));

		// Map z to the value
		sum_prob = 0;
		for (int i=1; i<=n; i++) {
			sum_prob = sum_prob + c / pow((double) i, alpha);
			if (sum_prob >= z) {
			  zipf_value = i;
			  break;
			}
		};

		// Assert that zipf_value is between 1 and N
		assert((zipf_value >=1) && (zipf_value <= n));

		return(zipf_value-1);
	}

	double rand_val()
	{
		// static long x;               // Random int value
		long        x_div_q;         // x divided by q
		long        x_mod_q;         // x modulo q
		long        x_new;           // New x value

		// RNG using integer arithmetic
		x_div_q = x / q;
		x_mod_q = x % q;
		x_new = (a * x_mod_q) - (r * x_div_q);
		if (x_new > 0)
		x = x_new;
		else
		x = x_new + m;

		// Return a random value between 0.0 and 1.0
		return((double) x / m);
	};

	void generateZipf() {
	  	int zipf_rv;               // Zipf random variable

	  	zipf_rv = zipf();

	  	// zipf_rv = n - zipf_rv; //n-1 to 0 (n-1 with highest probability)

	  	if (n <= 7) { //possibility #1 (year predicate)
	  		assert(range < 7);
	  		year = make_pair(1998 - zipf_rv - range, 1998 - zipf_rv);
	  		yearmonth = make_pair(year.first * 100 + 1, year.second * 100 + 12);
	  		date = make_pair(yearmonth.first * 100 + 1, yearmonth.second * 100 + 30);
	  	} else if (n <= 79) { //possibility #2 (yearmonthnum predicate)
	  		assert(range == 0);
	  		if (zipf_rv < 7) { // 0 to 6
		  		int temp = zipf_rv / 12;
		  		//zipf_rv = 0 to 6;
		  		//min = 0; 1998
		  		//max = 6; 1992
		  		year = make_pair(1998 - temp, 1998 - temp);
		  		temp = zipf_rv % 12;
		  		//min = 0; 12
		  		//max = 11; 1
		  		yearmonth = make_pair(year.first * 100 + 7 - temp - range, year.second * 100 + 7 - temp);
		  		date = make_pair(yearmonth.first * 100 + 1, yearmonth.second * 100 + 30);
	  		} else {
	  			zipf_rv -= 7;
		  		int temp = zipf_rv / 12;
		  		//zipf_rv = 0 to 71 or 7 to 78
		  		//min = 0; 1998
		  		//max = 6; 1992
		  		year = make_pair(1997 - temp, 1997 - temp);
		  		temp = zipf_rv % 12;
		  		//min = 0; 12
		  		//max = 11; 1
		  		yearmonth = make_pair(year.first * 100 + 12 - temp - range, year.second * 100 + 12 - temp);
		  		date = make_pair(yearmonth.first * 100 + 1, yearmonth.second * 100 + 30);
	  		}
	  	} else if (n <= 316) { //possibility #3 (week predicate)
	  		assert(range == 0);
	  		if (zipf_rv < 28) { // 0 to 27
				int temp = zipf_rv / 48;
				//zipf_rv = 0 to 27;
				//min = 0; 1998
				//max = 6; 1992
				year = make_pair(1998 - temp, 1998 - temp);
				temp = (zipf_rv % 48) / 4;
				//min = 0; 12
				//max = 11; 1
				yearmonth = make_pair(year.first * 100 + 7 - temp, year.second * 100 + 7 - temp);
				temp = (zipf_rv % 48) % 4;
				//min = 0; 22 - 28
				//max = 3; 1 - 7
				date = make_pair(yearmonth.first * 100 + 22 - temp * 7 - range * 7, yearmonth.second * 100 + 28 - temp * 7);
	  		} else {
				int temp = zipf_rv / 48;
				//zipf_rv = 0 to 27;
				//min = 0; 1998
				//max = 6; 1992
				year = make_pair(1997 - temp, 1997 - temp);
				temp = (zipf_rv % 48) / 4;
				//min = 0; 12
				//max = 11; 1
				yearmonth = make_pair(year.first * 100 + 12 - temp, year.second * 100 + 12 - temp);
				temp = (zipf_rv % 48) % 4;
				//min = 0; 22 - 28
				//max = 3; 1 - 7
				date = make_pair(yearmonth.first * 100 + 22 - temp * 7 - range * 7, yearmonth.second * 100 + 28 - temp * 7);
	  		}


	  	}

	  

	};
};

class Operator {
public:
	DeviceType device;
	int table_id;
	OperatorType type;
	unsigned short sg;
	short* segment_group;
	Operator* children;
	Operator* parents;

	vector<ColumnInfo*> columns;
	vector<ColumnInfo*> supporting_columns;

	Operator(DeviceType _device, unsigned short _sg, int _table_id, OperatorType _type) {
		type = _type;
		sg = _sg;
		table_id = _table_id;
		device = _device;
	};
	void addChild(Operator* child) {
		children = child;
		if (child != NULL) child->parents = this;
	};
	void setDevice(DeviceType _device) {
		device = _device;
	};

};

class QueryOptimizer {
public:
	CacheManager* cm;
	CPUGPUProcessing* cgp;

	vector<ColumnInfo*> querySelectColumn;
	vector<ColumnInfo*> queryBuildColumn;
	vector<ColumnInfo*> queryProbeColumn;
	vector<ColumnInfo*> queryGroupByColumn;
	vector<ColumnInfo*> queryAggrColumn;

	vector<pair<ColumnInfo*, ColumnInfo*>> join;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> aggregation;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> groupby_build;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> select_probe;
	unordered_map<ColumnInfo*, vector<ColumnInfo*>> select_build;

	unordered_map<ColumnInfo*, ColumnInfo*> fkey_pkey;
	unordered_map<ColumnInfo*, ColumnInfo*> pkey_fkey;

	vector<vector<vector<vector<Operator*>>>> opGPUPipeline; // for each table, for each segment group, for each pipeline, there is vector of operator
	vector<vector<vector<vector<Operator*>>>> opCPUPipeline; // for each table, for each segment group, for each pipeline, there is vector of operator

	vector<vector<Operator*>> opRoots; // for each table, for each segment group there is operator

	vector<vector<Operator*>> opParsed; // for each table, there is vector of operator

	vector<vector<ColumnInfo*>> joinGPUPipelineCol;
	vector<vector<ColumnInfo*>> joinCPUPipelineCol;
	vector<vector<ColumnInfo*>> selectGPUPipelineCol;
	vector<vector<ColumnInfo*>> selectCPUPipelineCol;
	vector<vector<ColumnInfo*>> groupbyGPUPipelineCol;
	vector<vector<ColumnInfo*>> groupbyCPUPipelineCol;

	vector<vector<ColumnInfo*>> queryColumn;

	bool groupGPUcheck;
	bool joinGPUall;
	bool* joinGPUcheck, *joinCPUcheck, **joinGPU, **joinCPU;

	short** segment_group, **segment_group_temp;
	short** segment_group_count, **segment_group_temp_count;
	short** par_segment;
	short* par_segment_count;
	int* last_segment;

	map<int, map<ColumnInfo*, double>> speedup;
	double** speedup_segment;
	map<int, Zipfian*> zipfian;
	QueryParams* params;

	QueryOptimizer(size_t _cache_size, size_t _ondemand_size, size_t _processing_size, size_t _pinned_memsize, CPUGPUProcessing* _cgp);
	~QueryOptimizer();

	void parseQuery(int query);
	void parseQuery11();
	void parseQuery12();
	void parseQuery13();
	void parseQuery21();
	void parseQuery22();
	void parseQuery23();
	void parseQuery31();
	void parseQuery32();
	void parseQuery33();
	void parseQuery34();
	void parseQuery41();
	void parseQuery42();
	void parseQuery43();

	void prepareQuery(int query, bool skew = 0);

	void clearParsing();
	void clearPlacement();
	void clearPrepare();

	void dataDrivenOperatorPlacement(int query, bool isprofile = 0);
	void prepareOperatorPlacement();
	void groupBitmap(bool isprofile = 0);
	void groupBitmapSegment(int query, bool isprofile = 0);
	void groupBitmapSegmentTable(int table_id, int query, bool isprofile = 0);
	void groupBitmapSegmentTableOD(int table_id, int query, bool isprofile = 0);

	bool checkPredicate(int table_id, int segment_idx);
	void updateSegmentStats(int table_id, int segment_idx, int query);

};

#endif