#include "test_compile.h"

Segment::Segment(ColumnInfo* _column, int* _seg_ptr, int _priority)
: column(_column), seg_ptr(_seg_ptr), priority(_priority), seg_size(SEGMENT_SIZE) {
	stats = new Statistics();
	col_ptr = column->col_ptr;
	segment_id = (seg_ptr - col_ptr)/seg_size;
	weight = 0;
}

Segment::Segment(ColumnInfo* _column, int* _seg_ptr)
: column(_column), seg_ptr(_seg_ptr), priority(0), seg_size(SEGMENT_SIZE) {
	stats = new Statistics();
	col_ptr = column->col_ptr;
	segment_id = (seg_ptr - col_ptr)/seg_size;
	weight = 0;
}

ColumnInfo::ColumnInfo(string _column_name, string _table_name, int _LEN, int _column_id, int _table_id, int* _col_ptr)
: column_name(_column_name), table_name(_table_name), LEN(_LEN), column_id(_column_id), table_id(_table_id), col_ptr(_col_ptr) {
	stats = new Statistics();
	tot_seg_in_GPU = 0;
	weight = 0;
	seg_ptr = col_ptr;
	total_segment = (LEN+SEGMENT_SIZE-1)/SEGMENT_SIZE;
}

Segment*
ColumnInfo::getSegment(int index) {
	Segment* seg = new Segment(this, col_ptr+SEGMENT_SIZE*index);
	return seg;
}