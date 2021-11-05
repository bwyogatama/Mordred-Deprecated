#include "QueryOptimizer.h"
#include "CostModel.h"

CostModel::CostModel(int _L, int _n_group_key, int _n_aggr_key, int _sg, int _table_id, QueryOptimizer* _qo) {
		L = _L;
		n_group_key = _n_group_key;
		n_aggr_key = _n_aggr_key;

		sg = _sg;
		table_id = _table_id;

		qo = _qo;


		Operator* op = qo->opRoots[table_id][sg];
		opPipeline.push_back(op);
		op = op->children;

		while (op != NULL) {
			if (op->type != Materialize && op->type != GPUtoCPU && op->type != CPUtoGPU && op->type != Merge) {
				opPipeline.push_back(op);
			}
			op = op->children;
		}
};

void 
CostModel::clear() {
	selectCPU.clear(); joinCPU.clear(); groupCPU.clear(); buildCPU.clear();
	selectGPU.clear(); joinGPU.clear(); groupGPU.clear(); buildGPU.clear();
}

void
CostModel::permute_cost() {
	for (int i = 0; i < opPipeline.size(); i++) {
		Operator* op = opPipeline[i];
		if (op->device == GPU) {
			if (op->type == Probe) joinGPU.push_back(op->columns[0]);
			else if (op->type == Filter) selectGPU.push_back(op->columns[0]);
			else if (op->type == GroupBy || op->type == Aggr) {
				for (int k = 0; k < op->columns.size(); k++)
					groupGPU.push_back(op->columns[k]);
			} else if (op->type == Build) buildGPU.push_back(op->columns[0]);
		} else {
			if (op->type == Probe) joinCPU.push_back(op->columns[0]);
			else if (op->type == Filter) selectCPU.push_back(op->columns[0]);
			else if (op->type == GroupBy || op->type == Aggr) {
				for (int k = 0; k < op->columns.size(); k++)
					groupCPU.push_back(op->columns[k]);
			} else if (op->type == Build) buildCPU.push_back(op->columns[0]);		
		}
	}

	double default_cost = calculate_cost();

	clear();

	for (int i = 0; i < opPipeline.size(); i++) {
		double cost = 0;

		Operator* cur_op = opPipeline[i];
		if (cur_op->device == CPU) cur_op->device = GPU;
		else if (cur_op->device == GPU) cur_op->device = CPU;

		for (int j = 0; j < opPipeline.size(); j++) {
			Operator* op = opPipeline[j];

			if (op->device == GPU) {
				if (op->type == Probe) joinGPU.push_back(op->columns[0]);
				else if (op->type == Filter) selectGPU.push_back(op->columns[0]);
				else if (op->type == GroupBy || op->type == Aggr) {
					for (int k = 0; k < op->columns.size(); k++)
						groupGPU.push_back(op->columns[k]);
				} else if (op->type == Build) buildGPU.push_back(op->columns[0]);	
			} else {
				if (op->type == Probe) joinCPU.push_back(op->columns[0]);
				else if (op->type == Filter) selectCPU.push_back(op->columns[0]);
				else if (op->type == GroupBy || op->type == Aggr) {
					for (int k = 0; k < op->columns.size(); k++)
						groupCPU.push_back(op->columns[k]);
				} else if (op->type == Build) buildCPU.push_back(op->columns[0]);			
			}
		}

		cost = calculate_cost();

		if (cur_op->device == CPU) {
			cur_op->device = GPU;
			for (int k = 0; k < cur_op->columns.size(); k++) {
				// speedup_segment[cur_op->columns[k]] = cost - default_cost;
			}
			for (int k = 0; k < cur_op->supporting_columns.size(); k++) {
				// speedup_segment[cur_op->supporting_columns[k]] += cost - default_cost;
			}
		} else if (cur_op->device == GPU) {
			cur_op->device = CPU;
			for (int k = 0; k < cur_op->columns.size(); k++) {
				// speedup_segment[cur_op->columns[k]] = default_cost - cost;
			}
			for (int k = 0; k < cur_op->supporting_columns.size(); k++) {
				// speedup_segment[cur_op->supporting_columns[k]] += default_cost - cost;
			}
		}

		clear();
	}

}

double 
CostModel::calculate_cost() {
	double cost = 0;

	bool fromGPU = false;
	if (selectGPU.size() > 0 || joinGPU.size() > 0) {
		for (int i = 0; i < selectGPU.size(); ++i) {	
			ColumnInfo* col = selectGPU[i];
			L *= qo->params->selectivity[col];
		}
		for (int i = 0; i < joinGPU.size(); ++i) {	
			ColumnInfo* col = joinGPU[i];
			L *= qo->params->selectivity[col];
		}
		cost += transfer_cost(joinGPU.size() + 1);
		fromGPU = true;
	}

	for (int i = 0; i < selectCPU.size(); i++) {
		ColumnInfo* col = selectCPU[i];
		if (fromGPU) {
			cost += filter_cost(qo->params->selectivity[col], 1, 0);
			fromGPU = false;
		} else cost += filter_cost(qo->params->selectivity[col], 0, 0);
	}

	for (int i = 0; i < joinCPU.size(); i++) {
		ColumnInfo* col = joinCPU[i];
		if (fromGPU) {
			cost += probe_cost(qo->params->selectivity[col], 1, 0);
			fromGPU = false;
		} else cost += probe_cost(qo->params->selectivity[col], 0, 0);
	}

	for (int i = 0; i < groupCPU.size(); i++) {
		ColumnInfo* col = groupCPU[i];
		if (fromGPU){
			cost += group_cost(0);
			fromGPU = false;
		} else cost += group_cost(1);
	}

	for (int i = 0; i < buildCPU.size(); i++) {
		ColumnInfo* col = buildCPU[i];
		if (fromGPU) {
			cost += build_cost(1);
			fromGPU = false;
		} else cost += build_cost(0);
	}

	return cost;

}

double 
CostModel::probe_cost(double selectivity, bool mat_start, bool mat_end) {

	double cost = 0;
	double scan_time = 0, probe_time = 0, write_time = 0;

	if (mat_start) scan_time = L * 4/BW_CPU + L * 4 * CACHE_LINE/BW_CPU;
	else scan_time = L * 4/BW_CPU;

	probe_time = L * 4 * CACHE_LINE/BW_CPU;

	if (mat_end) write_time = L * 4 * selectivity * 2/BW_CPU;
	else write_time = 0;

	L *= selectivity;

	cost = scan_time + probe_time + write_time;

	return cost;
}

double 
CostModel::transfer_cost(int M) {
	double transfer_time = L * 4 * M/BW_PCI;
	return transfer_time;
}

double 
CostModel::filter_cost(double selectivity, bool mat_start, bool mat_end) {

	double cost = 0;
	double scan_time = 0, write_time = 0;

	if (mat_start) scan_time = L * 4/BW_CPU + L * 4 * CACHE_LINE/BW_CPU;
	else scan_time = L * 4/BW_CPU;

	if (mat_end) write_time = L * 4 * selectivity/BW_CPU;
	else write_time = 0;

	L *= selectivity;

	cost = scan_time + write_time;

	return cost;
}

double 
CostModel::group_cost(bool mat_start) {

	double cost = 0;
	double scan_time = 0, group_time = 0;

	if (mat_start) scan_time = L * 4 * (n_group_key + 1)/BW_CPU + L * 4 * CACHE_LINE * (n_aggr_key + n_group_key)/BW_CPU;
	else scan_time = L * 4 * CACHE_LINE * n_aggr_key/BW_CPU;

	group_time = L * 4 * CACHE_LINE/BW_CPU;

	cost = scan_time + group_time;

	return cost;
}

double 
CostModel::build_cost(bool mat_start) {

	double cost = 0;
	double scan_time = 0, build_time = 0;

	if (mat_start) scan_time = L * 4/BW_CPU + L * 4 * CACHE_LINE/BW_CPU;
	else scan_time = L * 4/BW_CPU;

	build_time = L * 4 * CACHE_LINE/BW_CPU;

	cost = scan_time + build_time;

	return cost;
}