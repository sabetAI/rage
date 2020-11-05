deepspeed rag_train.py ~/gec-data/merged/bea.train.dev.tsv rag --batch_size 4 --num_workers 4 \
    --deepspeed --deepspeed_config ds_config.json
