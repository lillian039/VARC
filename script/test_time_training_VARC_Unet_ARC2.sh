file_names=('65b59efc' '2d0172a1' '7b0280bc' 'e12f9a14' 'e8686506' '88e364bc' '7b3084d4' 'a251c730' 'd35bdbdc' 'fc7cae8d' '8f3a5a89' '35ab12c3' '88bcf3b4' 'cb2d8a2c' 'b10624e5' '135a2760' '5dbc8537' 'e87109e9' '4a21e3da' 'abc82100' '64efde09' '3e6067c3' '2ba387bc' '5961cc34' '38007db0' '20270e3b' '142ca369' '67e490f4' 'bf45cf4b' 'dbff022c' '4e34c42c' '36a08778' '7c66cb00' 'a395ee82' '271d71e2' 'f931b4a8' 'faa9f03d' '581f7754' 'a25697e4' '71e489b6' '21897d95' '8698868d' '1818057f' '6e4f6532' '9385bd28' '0934a4d8' '4c3d4a41' '20a9e565' 'db695cfb' '78332cb0' '80a900e0' 'a47bf94d' '800d221b' 'aa4ec2a5' 'a6f40cea' '13e47133' 'f560132c' '8b9c3697' '8f215267' '247ef758' 'de809cff' 'b99e7126' 'd8e07eb2' '62593bfd' 'e376de54' '16de56c4' '4c7dc4dd' '7491f3cf' 'b9e38dc0' '221dfab4' '332f06d7' '45a5af55' 'eee78d87' '3dc255db' 'a32d8b75' '4c416de3' 'edb79dae' '136b0064' 'b5ca7ac4' '6e453dd6' '7b80bb43' 'b6f77b65' '6ffbe589' 'dd6b8c4b' '28a6681f' '7666fa5d' '2c181942' '7b5033c1' 'c4d067a0' 'da515329' '446ef5d2' '269e22fb' '409aa875' 'e3721c99' '291dc1e1' 'db0c5428' '8b7bacbf' '58490d8a' '5545f144' '2b83f449' '9aaea919' 'dfadab01' '89565ca0' '53fb4810' '31f7f899' '58f5dbd5' '1ae2feb7' 'b0039139' 'c7f57c3e' '3a25b0d8' '16b78196' '9bbf930d' '898e7135' '7ed72f31' 'cbebaa4b' '97d7923e' 'd59b0160' '8e5c0c38' '981571dc' '195c6913')
NUM_GPUS=8

for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
  (
    i=0
    for file_name in "${file_names[@]}"; do
      # Shard tasks by index: each GPU gets roughly 1/8 of the file_names
      if (( i % NUM_GPUS == gpu )); then
        echo "GPU ${gpu} processing task ${file_name} (ARC-2 VARC-Unet)"
        CUDA_VISIBLE_DEVICES=${gpu} python test_time_train_ARC.py \
            --epochs 100 \
            --unet-size big \
            --batch-size 8 \
            --image-size 64 \
            --patch-size 2 \
            --learning-rate 3e-4 \
            --weight-decay 0 \
            --embed-dim 512 \
            --num-colors 12 \
            --resume-checkpoint "saves/offline_train_Unet/checkpoint_best.pt" \
            --lr-scheduler "cosine" \
            --train-split "eval_color_permute_ttt_9/${file_name}" \
            --data-root "raw_data/ARC-AGI-2" \
            --eval-split "eval_color_permute_ttt_9/${file_name}" \
            --resume-skip-task-token \
            --architecture "unet" \
            --eval-save-name "ARC_2_eval_Unet" \
            --num-attempts 10 \
            --ttt-num-each 2
      fi
      ((i++))
    done
  ) &
done

wait
echo "All VARC-Unet ARC-2 tasks finished."