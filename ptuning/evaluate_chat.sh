PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000

CHAT_TRAIN_DATA="/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.train.json"
CHAT_VAL_DATA="/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.val.json"
CHATGLM_MODEL="/aistudio/workspace/system-default/envs/python3.8/software/test_0331/hug_models/chatglm-6b"
CHECKPOINT_NAME="/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/checkpoint-ptuning-v3"

CUDA_VISIBLE_DEVICES=0 /aistudio/workspace/system-default/envs/python3.8/bin/python  main.py \
    --do_predict \
    --validation_file $CHAT_VAL_DATA \
    --test_file $CHAT_VAL_DATA \
    --overwrite_cache \
    --prompt_column prompt \
    --model_name_or_path $CHATGLM_MODEL \
    --ptuning_checkpoint /aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/checkpoint-ptuning-v3/checkpoint-3000 \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
