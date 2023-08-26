SF=1

# Make sure encoder is using right scale factor
bin=bin/gpudb/minmax
binsort=bin/gpudb/minmaxsort

LO_LEN=6001171
P_LEN=200000
S_LEN=2000
C_LEN=30000
D_LEN=2556

# arr=("lo_custkey" "lo_partkey" "lo_suppkey" "lo_orderdate" "lo_quantity" "lo_extendedprice" "lo_discount" "lo_revenue" "lo_supplycost" "lo_orderkey" "lo_linenumber" "lo_tax" "lo_ordtotalprice" "lo_commitdate")
# for val in ${arr[*]}; do
#  echo $val
#  $bin $val $LO_LEN
# done

arr=("lo_custkey" "lo_partkey" "lo_suppkey" "lo_orderdate" "lo_quantity" "lo_extendedprice" "lo_discount" "lo_revenue" "lo_supplycost" "lo_orderkey" "lo_linenumber" "lo_tax" "lo_ordtotalprice" "lo_commitdate")
for val in ${arr[*]}; do
 echo $val
 $binsort $val $LO_LEN $SF
done

arr=("p_partkey" "p_mfgr" "p_category" "p_brand1")
for val in ${arr[*]}; do
 echo $val
 $bin $val $P_LEN $SF
done

arr=("c_custkey" "c_region" "c_nation" "c_city")
for val in ${arr[*]}; do
 echo $val
 $bin $val $C_LEN $SF
done

arr=("s_suppkey" "s_region" "s_nation" "s_city")
for val in ${arr[*]}; do
 echo $val
 $bin $val $S_LEN $SF
done

arr=("d_datekey" "d_year" "d_yearmonthnum")
for val in ${arr[*]}; do
 echo $val
 $bin $val $D_LEN $SF
done
