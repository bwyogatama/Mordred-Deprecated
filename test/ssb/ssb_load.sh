export SAMPLE_PATH="/home/ubuntu/Implementation-GPUDB/test/ssb/data/s160"
export MAPD_DATA="data/mapd_import/ssb/"
export QUERIES="/home/ubuntu/Implementation-GPUDB/test/ssb/queries/transformed"

./bin/omnisci_server --num-gpus=1 --enable-cpu-sub-tasks=0

./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274 < /home/ubuntu/Implementation-GPUDB/test/ssb/ssb_table2.sql

rm $MAPD_DATA/*
ln $SAMPLE_PATH/customer.tbl.p $MAPD_DATA/customer.tbl.p
ln $SAMPLE_PATH/part.tbl.p $MAPD_DATA/part.tbl.p
ln $SAMPLE_PATH/supplier.tbl.p $MAPD_DATA/supplier.tbl.p
ln $SAMPLE_PATH/date.tbl $MAPD_DATA/date.tbl
ln $SAMPLE_PATH/lineorder.tbl $MAPD_DATA/lineorder.tbl

echo "copy customer from '$MAPD_DATA/customer.tbl.p' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy part from '$MAPD_DATA/part.tbl.p' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy supplier from '$MAPD_DATA/supplier.tbl.p' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy ddate from '$MAPD_DATA/date.tbl' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274
echo "copy lineorder from '$MAPD_DATA/lineorder.tbl' with (delimiter='|');" | ./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274

./bin/omnisql omnisci -u admin -p HyperInteractive --port 6274

#COGADB
sudo docker images --all
sudo docker ps
sudo docker run --rm --it nvidia/cuda:cogadb
sudo docker commit container_id new_image_name

sudo docker login -u bwyogatama docker.io
sudo docker tag nvidia/cuda:cogadb bwyogatama/cogadb:10.0-devel-ubuntu14.04
sudo docker push bwyogatama/cogadb:10.0-devel-ubuntu14.04