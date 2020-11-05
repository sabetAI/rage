deepspeed rage_train.py ~/gec-data/merged/bea.train.dev.tsv rag --num_workers 4 \
    --deepspeed --deepspeed_config ds_config.json
