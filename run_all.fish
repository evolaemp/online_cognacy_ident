#!/usr/bin/fish

begin
	set -x PYTHONHASHSEED 42

	set -l datasets_dir 'datasets'
	set -l output_dir 'output'

	set -l ipa_datasets abvd bai chinese_1964 chinese_2004 \
						ielex japanese ob_ugrian tujia

	mkdir -p "$output_dir/pmi"
	mkdir -p "$output_dir/phmm"

	for dataset_path in (find datasets -type f -name '*.tsv' | sort)
		set -l dataset_name (basename $dataset_path | cut -d '.' -f 1)

		set -l ipa_flag
		if contains $dataset_name $ipa_datasets
			set ipa_flag '--ipa'
		end

		python run.py pmi $dataset_path $ipa_flag --evaluate \
			--output "$output_dir/pmi/$dataset_name.tsv"
		python run.py phmm $dataset_path $ipa_flag --evaluate \
			--output "$output_dir/phmm/$dataset_name.tsv"

		echo
	end
end
