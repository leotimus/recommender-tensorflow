#!/usr/bin/env bash

#SBATCH --job-name ModelTest
#SBATCH --gres=gpu:1

modelSpec=false
dataSpech=false
read -p "Username:" uname
read -s -p "Password:" psd
while [[ $# -gt 0 ]] && [[ $1 == "-"* ]]
do
	flag=$1
	shift
	case $flag in
		"-m" )
			model=$1
			modelSpec=true
			shift ;;
		"--loss" )
			param="$param loss $1"
			shift ;;
		"-o" )
			param="$param opti $1"
			shift ;;
		"-d" )
			param="$param data $1"
			dataSpech=true
			shift ;;
		"-e" ) 
			param="$param epoch $1"
			shift ;;
		"--lrate" )
			param="$param lrate $1"
			shift ;;
		"-r" )
			param="$param ratio $1"
			shift ;;
		"-k" ) param="$param k $1"
			shift ;;

	esac
done

if ! $modelSpec
then 
	echo "ERROR - the model to train was not given. Specify the model name after the flag -m"
	exit 2
fi

#if ! $dataSpech
#then 
#	echo "ERROR - the data to train the model on were not given. Specify the data path after the flag -d"
#	exit 2
#fi
echo $param
srun --gres=gpu:1 --job-name=twoTower --mem 128G singularity exec --nv environement.sif ./evalModel.sh $model $uname $psd $param

exit 0
