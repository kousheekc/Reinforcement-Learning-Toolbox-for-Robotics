#!/bin/bash

for seed in {0,1,2,3,4,5,6,7,8,9}
do
	python3 -m rlbotics.td3.main --seed $seed
done
