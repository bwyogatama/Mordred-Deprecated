LOG_DIR=/home/ubuntu/CS764-GPUCompression/logs/
SF=20

mkdir -p $LOG_DIR/ssb/$SF/

bin/cpu/ssb/q11 --t=3 > $LOG_DIR/ssb/$SF/q11cpu
bin/cpu/ssb/q12 --t=3 > $LOG_DIR/ssb/$SF/q12cpu
bin/cpu/ssb/q13 --t=3 > $LOG_DIR/ssb/$SF/q13cpu
bin/cpu/ssb/q21 --t=3 > $LOG_DIR/ssb/$SF/q21cpu
bin/cpu/ssb/q22 --t=3 > $LOG_DIR/ssb/$SF/q22cpu
bin/cpu/ssb/q23 --t=3 > $LOG_DIR/ssb/$SF/q23cpu
bin/cpu/ssb/q31 --t=3 > $LOG_DIR/ssb/$SF/q31cpu
bin/cpu/ssb/q32 --t=3 > $LOG_DIR/ssb/$SF/q32cpu
bin/cpu/ssb/q33 --t=3 > $LOG_DIR/ssb/$SF/q33cpu
bin/cpu/ssb/q34 --t=3 > $LOG_DIR/ssb/$SF/q34cpu
bin/cpu/ssb/q41 --t=3 > $LOG_DIR/ssb/$SF/q41cpu
bin/cpu/ssb/q42 --t=3 > $LOG_DIR/ssb/$SF/q42cpu
bin/cpu/ssb/q43 --t=3 > $LOG_DIR/ssb/$SF/q43cpu