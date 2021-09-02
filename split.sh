#! /bin/bash


dirs=$(ls -d */)
TRAIN_SZ=186
VAL_SZ=40
TEST_SZ=40

for freq_dir in $dirs
do
	mkdir -p $freq_dir/{train,validation,test}
	for ((i=1;i<=TRAIN_SZ;i++)); do
		mv $freq_dir/$i.csv $freq_dir/train
	done

	upLimit=$((TRAIN_SZ+VAL_SZ))
	lowLimit=$((TRAIN_SZ+1))

	for ((i=lowLimit, j=1;i<=upLimit;i++, j++)); do
		mv $freq_dir/$i.csv $freq_dir/validation/$j.csv
	done

	upLimit=$((upLimit+TEST_SZ))
	lowLimit=$((lowLimit+VAL_SZ))

	for ((i=lowLimit, j=1;i<=upLimit;i++, j++)); do
		mv $freq_dir/$i.csv $freq_dir/test/$j.csv
	done

done


