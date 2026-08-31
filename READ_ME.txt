Week 1:
    pathway will include running train.py, generate.py, & compare_to_real.py
    commands for this in terminal (I use python3 as I'm on macOS):

    python3 src/train.py --checkpoint checkpoints/lstm_best.pt
    python3 src/generate.py --checkpoint checkpoints/lstm_best.pt
    python3 src/compare_to_real.py --generated outputs/generated_XXXX.jsonl --real data/real_sequences.csv --pwms data/jaspar_pwms.txt

#for next week: get more sequences from Tisha, find more KRAS pathway motif families, make the above process occur w/o manual input

python3 src/compare_to_real.py --generated outputs/gen_c_label_1787153926.jsonl --real data/real_sequences.csv --pwms data/jaspar_pwms.txt



Week 2:
    Model V1 integrated into QC. Train.py, generate.py, & validate.py scripts
    all into pipeline.py.

    Entrypoint:
    python src/pipeline.py --config configs/baseline.yaml

#for next week: fix debugging, compare enformer vs. simple rep, fix class imbalance, 

Week 3:
Worked on checkpoint loading mismatch (Markov vs. LSTM), QC length-check hardcoded to old spec, motif-matching threshold bug for HNF4G.
    Entrypoint:
    python3 src/pipeline.py --config configs/v1.yaml

    Reproducible Manifest: outputs/model_v1_manifest_1788143951.json
