#!/usr/bin/fish

set -x PYTHONHASHSEED 42

# path to the data all the models are trained on
set -g training_dataset 'training_data/asjpv17_word_pairs.txt'

# names of the datasets that need the --ipa flag
set -g ipa_datasets abvd bai chinese_1964 chinese_2004 \
					ielex japanese ob_ugrian tujia


# train a model with certain hyperparameters, apply the model on a dataset and
# evaluate the output against that dataset
function train_run_eval
	set dataset $argv[1]
	set algo $argv[2]
	set batch_size $argv[3]
	set alpha $argv[4]

	set ipa_flag
	if contains $dataset $ipa_datasets
		set ipa_flag '--ipa'
	end

	set model_name "models/$algo/asjp_m=$batch_size,α=$alpha.$algo"

	if not test -f $model_name
		python train.py $algo \
			$training_dataset \
			$model_name \
			--batch-size $batch_size --alpha $alpha \
			--time
	end

	python run.py $model_name \
		datasets/$dataset.tsv $ipa_flag \
		--output output/$algo/$dataset.tsv \
		--evaluate

	echo
end


# make sure these dirs exist
mkdir -p models/pmi models/phmm
mkdir -p output/pmi output/phmm


# train+run+eval all datasets with the hyperparameters of table 4 of the paper
train_run_eval abvd pmi 64 0.75
train_run_eval abvd phmm 32 0.5

train_run_eval afrasian pmi 256 0.65
train_run_eval afrasian phmm 32 0.8

train_run_eval bai pmi 8192 0.75
train_run_eval bai phmm 32 0.55

train_run_eval chinese_1964 pmi 128 0.95
train_run_eval chinese_1964 phmm 512 0.6

train_run_eval chinese_2004 pmi 128 0.95
train_run_eval chinese_2004 phmm 512 0.6

train_run_eval huon pmi 32 1
train_run_eval huon phmm 32 0.65

train_run_eval ielex pmi 512 0.55
train_run_eval ielex phmm 1024 0.5

train_run_eval japanese pmi 512 0.55
train_run_eval japanese phmm 32 0.6

train_run_eval kadai pmi 2048 0.7
train_run_eval kadai phmm 32 0.7

train_run_eval kamasau pmi 512 0.5
train_run_eval kamasau phmm 128 0.55

train_run_eval lolo_burmese pmi 16384 0.5
train_run_eval lolo_burmese phmm 32 0.75

train_run_eval mayan pmi 64 0.5
train_run_eval mayan phmm 32 0.55

train_run_eval miao_yao pmi 8192 0.95
train_run_eval miao_yao phmm 128 0.7

train_run_eval mixe_zoque pmi 256 0.7
train_run_eval mixe_zoque phmm 32 0.7

train_run_eval mon_khmer pmi 256 0.7
train_run_eval mon_khmer phmm 32 0.5

train_run_eval ob_ugrian pmi 512 0.75
train_run_eval ob_ugrian phmm 32768 0.5

train_run_eval tujia pmi 1024 0.65
train_run_eval tujia phmm 32 0.5
