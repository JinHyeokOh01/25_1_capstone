for seq_len in 336
do
    for dp in 0 0.25 0.5 0.6
    do
        for pred_len in 96 192 336 720
        do
            echo ETTm2_${seq_len}-${pred_len}_${dp}
            python singlecard.py \
                --data ETTm2 \
                --data_path ETTm2.csv \
                --features M \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --enc_hidden 336 \
                --levels 2 \
                --lr 1e-4 \
                --dropout ${dp} \
                --batch_size 512 \
                --RIN 1 \
                --model_name DPAD_SE_ETTm2_I${seq_len}_o${pred_len}_lr1e-4_bs512_dp${dp}_h336_l2 \
                --model DPAD_SE
        done
    done
done

