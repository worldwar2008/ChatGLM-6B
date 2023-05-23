PRE_SEQ_LEN=128
LR=1e-2
CHAT_TRAIN_DATA="/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.train.json"
CHAT_VAL_DATA="/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.val.json"
CHATGLM_MODEL="/aistudio/workspace/system-default/envs/python3.8/software/test_0331/hug_models/chatglm-6b"
CHECKPOINT_NAME="/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/checkpoint-ptuning-v3"
CUDA_VISIBLE_DEVICES=0,1,2,3 /aistudio/workspace/system-default/envs/python3.8/bin/python /aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/ptuning/main.py \
    --do_train \
    --train_file $CHAT_TRAIN_DATA \
    --validation_file $CHAT_VAL_DATA \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path $CHATGLM_MODEL \
    --output_dir $CHECKPOINT_NAME \
    --overwrite_output_dir \
    --max_source_length 384 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    > /aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/run.log

