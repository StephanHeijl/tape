import pandas
import subprocess


df = pandas.read_csv("/home/sheijl/Documents/model_run_list_pt.csv").fillna(False)
for r, row in df.iterrows():
    ps = []

    for i, task_type in enumerate(["melting_point_classification", "melting_point_regression"]):
        if row.tokenizer:
            tokenizer = row.tokenizer
        else:
            tokenizer = "unirep" if row.model_name == "unirep" else "iupac"
        i = 0
        #wandb_project = "tape-melt-mmseq-3"
        wandb_project = "tape_prottrans"
        cmd = f"WANDB_PROJECT={wandb_project} CUDA_VISIBLE_DEVICES={i} tape-train {row.model_name} {task_type} --output_dir melting_point_regr_mmseq_2 --batch_size {row.batch_size}"
        cmd += f" --gradient_accumulation_steps {row.gradient_accumulation_steps} --learning_rate {row.learning_rate} --num_train_epochs {row.num_train_epochs}"
        cmd += f" --warmup_steps 0 --num_workers 0 --save_freq improvement --model_config_file config/{row.config_file} --tokenizer {tokenizer}"
        if row.from_pretrained:
            cmd += f" --from_pretrained {row.from_pretrained}"
        print(cmd)
        ps.append(subprocess.Popen(cmd, shell=True))
        ps[-1].communicate()