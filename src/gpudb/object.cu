#include "CacheManager.h"

enum OperatorType {
    Filter, Probe, Build, GroupBy, CPUtoGPU, GPUtoCPU, Materialize, Merge
};

class Operator {
public:
	int device;
	OperatorType type;
	unsigned short sg;
	short* segment_group;
	vector<Operator*> children;
	Operator* parents;

	ColumnInfo* column;
	vector<ColumnInfo*> supporting_columns;

	Operator(OperatorType _type, unsigned short _sg) {
		type = _type;
		sg = _sg;
	};
	void appendChild(Operator* child) {
		children.push_back(child);
	};
	void setParent(Operator* parent) {
		parents = parent;
	};

};