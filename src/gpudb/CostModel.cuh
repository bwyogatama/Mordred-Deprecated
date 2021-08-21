#ifndef _COST_MODEL_H
#define _COST_MODEL_H

#define CACHE_LINE 64
#define BW_CPU 100000
#define BW_PCI 12000

class CostModel {
public:
	CostModel() {};

	double probe_cost(int L, double selectivity, int join_so_far) {

		double cost = 0;
		double scan_time = 0, probe_time = 0, write_time = 0, transfer_time = 0;


		scan_time = L * 4/BW_CPU + L * 4 * CACHE_LINE/BW_CPU;

		transfer_time = L * 4 * (join_so_far + 1)/BW_PCI + L * 4 * selectivity * (join_so_far + 2)/BW_PCI;

		write_time = L * 4 * selectivity * CACHE_LINE * 2/BW_CPU;

		probe_time = L * 4 * selectivity * CACHE_LINE/BW_CPU;

		cost = scan_time + probe_time + write_time + transfer_time;

		return cost;
	}

	double filter_cost(int L, double selectivity) {

		double cost = 0;
		double scan_time = 0, write_time = 0, transfer_time = 0;

		scan_time = L * 4/BW_CPU;

		transfer_time = L * 4 * selectivity/BW_PCI;

		write_time = L * 4 * selectivity * CACHE_LINE/BW_CPU;

		cost = scan_time + write_time + transfer_time;

		return cost;
	}

	double group_cost(int L, int join_so_far, int n_group_key, int n_aggr_key) {

		double cost = 0;
		double scan_time = 0, group_time = 0, transfer_time = 0;

		scan_time = L * 4 * (n_group_key + 1)/BW_CPU + L * 4 * CACHE_LINE * (n_aggr_key + n_group_key)/BW_CPU;

		transfer_time = L * 4 * (join_so_far + 1)/BW_PCI;

		group_time = L * 4 * CACHE_LINE * (n_group_key + 1)/BW_CPU;

		cost = scan_time + group_time + transfer_time;

		return cost;
	}

	double build_cost(int D, int L, double selectivity, bool isgroup = true) {

		double cost = 0;
		double scan_time = 0, build_time = 0, transfer_time = 0, probe_time = 0;

		int k;
		if (isgroup) k = 2;
		else k = 1;

		scan_time = D * 4/BW_CPU + D * 4 * CACHE_LINE * k/BW_CPU;

		transfer_time = D * 4/BW_PCI;

		build_time = D * 4 * CACHE_LINE * k/BW_CPU;

		probe_time = L * 4 * selectivity * CACHE_LINE/BW_CPU;

		cost = scan_time + transfer_time + build_time + probe_time;

		return cost;
	}
};

#endif