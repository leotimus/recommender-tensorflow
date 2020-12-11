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

echo $param
srun --gres=gpu:0 --job-name=SVD --mem 8G singularity exec --nv environement.sif ./evalModel.sh SVD $uname $psd $param

exit 0
