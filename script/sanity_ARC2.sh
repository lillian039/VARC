file_name="7666fa5d"

python test_time_train_ARC.py \
          --epochs 100 \
          --depth 10 \
          --batch-size 8 \
          --image-size 64 \
          --patch-size 2 \
          --learning-rate 3e-4 \
          --weight-decay 0 \
          --embed-dim 512 \
          --num-heads 8 \
          --num-colors 12 \
          --resume-checkpoint "saves/offline_train_ViT/checkpoint_best.pt" \
          --lr-scheduler "cosine" \
          --train-split "eval_color_permute_ttt_9/${file_name}" \
          --data-root "raw_data/ARC-AGI-2" \
          --eval-split "eval_color_permute_ttt_9/${file_name}" \
          --resume-skip-task-token \
          --architecture "vit" \
          --eval-save-name "ARC_2_eval_ViT_sanity" \
          --num-attempts 10 \
          --ttt-num-each 1