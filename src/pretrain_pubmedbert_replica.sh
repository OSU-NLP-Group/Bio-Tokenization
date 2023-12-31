deepspeed --master_port 7002 run_pretraining_automodel.py \
  --model_type bert-mlm \
  --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --tokenizer_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --lr 6e-4 \
  --train_batch_size 8192 \
  --train_micro_batch_size_per_gpu 1024 \
  --lr_schedule step \
  --curve linear \
  --warmup_proportion 0.10 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 480000.0 \
  --early_exit_time_marker 480000.0 \
  --max_steps 63000 \
  --dataset_path dataset/data/pubmedbert \
  --output_dir ../output/pubmedbert_replica_temp \
  --print_steps 1000 \
  --num_epochs_between_checkpoints 50 \
  --job_name pretraining_experiment \
  --project_name budget-bert-pretraining \
  --validation_epochs 6 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 1000000 \
  --early_stop_eval_loss 12 \
  --seed 42 \
  --fp16 \
  --layer_norm_type pytorch \
  --start_long_phase_perc 0.9

deepspeed --master_port 7003 --include="localhost:5" model_load.py \
  --model_type bert-mlm \
  --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --tokenizer_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --lr 6e-4 \
  --train_batch_size 16 \
  --train_micro_batch_size_per_gpu 16 \
  --lr_schedule step \
  --curve linear \
  --warmup_proportion 0.10 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 480000.0 \
  --early_exit_time_marker 480000.0 \
  --max_steps 10 \
  --dataset_path dataset/data/pubmedbert \
  --output_dir ../output/pubmedbert_replica \
  --print_steps 1000 \
  --num_epochs_between_checkpoints 50 \
  --job_name pretraining_experiment \
  --project_name budget-bert-pretraining \
  --validation_epochs 6 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 1000000 \
  --early_stop_eval_loss 12 \
  --seed 42 \
  --fp16 \
  --layer_norm_type pytorch \
  --start_long_phase_perc 0.9 \
  --load_training_checkpoint ../output/pubmedbert_replica_temp/pretraining_experiment-
  
rm -r ../output/pubmedbert_replica_temp