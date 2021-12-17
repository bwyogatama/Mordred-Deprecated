export SAMPLE_PATH="/home/ubuntu/Implementation-GPUDB/test/ssb/data/s160"
export MAPD_DATA="data/mapd_import/ssb/"
export QUERIES="/home/ubuntu/Implementation-GPUDB/test/ssb/queries/transformed"

./bin/omnisci_server --num-gpus=1 --enable-cpu-sub-tasks=1

./bin/omnisci_server --num-gpus=1 --enable-cpu-sub-tasks=1 --enable-watchdog=0

rm $MAPD_DATA/*
ln $SAMPLE_PATH/customer.tbl.p $MAPD_DATA/customer.tbl.p
ln $SAMPLE_PATH/part.tbl.p $MAPD_DATA/part.tbl.p
ln $SAMPLE_PATH/supplier.tbl.p $MAPD_DATA/supplier.tbl.p
ln $SAMPLE_PATH/date.tbl $MAPD_DATA/date.tbl
ln $SAMPLE_PATH/lineorder.tbl $MAPD_DATA/lineorder.tbl

./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274 < /home/ubuntu/Implementation-GPUDB/test/ssb/ssb_table2.sql

echo "copy customer from '$MAPD_DATA/customer.tbl.p' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy part from '$MAPD_DATA/part.tbl.p' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy supplier from '$MAPD_DATA/supplier.tbl.p' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy ddate from '$MAPD_DATA/date.tbl' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy lineorder from '$MAPD_DATA/lineorder.tbl' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274

./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274 < $QUERIES/q11.sql

#COGADB
#use script utility_scripts/install_cogadb_dependencies.sh from Hawk-VLDBJ
#don't install nvidia-cuda-toolkit
sudo docker images --all
sudo docker ps
sudo docker run -v ~/Implementation-GPUDB/test/ssb/data:/data --rm --gpus "device=0" -it bwyogatama/cogadb:6.5-devel-ubuntu14.04 

sudo docker commit container_id new_image_name

mkdir -p /cogadata
./cogadb/bin/cogadbd
set path_to_database=/cogadata
create_ssb_database /data/s160/
loaddatabase

select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount between 5 and 7 and lo_quantity between 26 and 35;

		
sudo docker login -u bwyogatama docker.io
sudo docker tag nvidia/cuda:cogadb bwyogatama/cogadb:6.5-devel-ubuntu14.04
sudo docker push bwyogatama/cogadb:6.5-devel-ubuntu14.04