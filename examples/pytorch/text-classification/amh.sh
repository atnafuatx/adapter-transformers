#export TASK_NAME=xnli2023
export CUDA_VISIBLE_DEVICES=1
export BASEPATH="/data/users/jalabi/slik1/code/Mad-x/adapter-transformers/examples/pytorch/text-classification/"
export MODELPATH="/data/users/jalabi/slik1/code/Mad-x/adapter-transformers/examples/pytorch/text-classification/models/"
DATADIR="/data/users/jalabi/slik1/code/Mad-x/data/xnli/nllb/de/"
for SEED in 1 2 3; do
	taskname=de_nllb
	outputdir="${MODELPATH}XNLI/baselines/NLLB/de/$SEED/"
	mkdir -p $outputdir
	python3 ${BASEPATH}run_text.py \
		--model_name_or_path xlm-roberta-base \
		--task_name xnli \
		--train_file ${DATADIR}train.tsv \
		--validation_file ${DATADIR}validation.tsv \
		--label_file labels.txt \
		--language de \
		--load_lang_adapter de/wiki@ukp \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--eval_steps 5000 \
		--per_device_train_batch_size 418 \
		--num_train_epochs 20 \
		--output_dir $outputdir \
		--overwrite_output_dir \
		--save_steps 20000 \
		--train_adapter \
		--evaluation_strategy steps \
		--adapter_config pfeiffer \
		--load_best_model_at_end \
		--save_total_limit 2 \
		--seed $SEED
done
