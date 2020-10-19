import os
import pandas
import json
import subprocess
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, f1_score
from scipy.stats import pearsonr
from tape.datasets import LMDBDataset


def load_real_values(lmdb_filename):
    ds = LMDBDataset(lmdb_filename)
    return {v['id']: float(v['target']) for v in ds}


def get_stats(fname, real_valid_values):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    df = pandas.DataFrame(data[1])
    preds = np.array(df.prediction.tolist())
    targets = np.array(df.target.tolist())

    if preds.shape[1] > 1:
        # Classification mode
        soft_targets = np.array([real_valid_values[item['id']] for item in data[1]])
        preds_i = np.argmax(preds, axis=1)
        epreds = np.exp(preds)
        epred_sums = epreds.sum(axis=1)
        epreds = (epreds.T / epred_sums).T
        epreds_f = (epreds * np.arange(0, 101, 5)).sum(axis=1)

        full_r = pearsonr(epreds_f, soft_targets)
        full_mae = mean_absolute_error(soft_targets, epreds_f)
        melters_r = pearsonr(epreds_f[targets < 19], soft_targets[targets < 19])
        melters_mae = mean_absolute_error(soft_targets[targets < 19], epreds_f[targets < 19])
        melters_f1 = f1_score(targets >= 19, preds_i >= 19)

    else:
        # Regression mode
        preds = preds.flatten()
        #         targets = targets.flatten()
        #
        #         pandas.Series(preds).hist()
        #         plt.show()
        #         pandas.Series(targets).hist()

        targets = np.array([real_valid_values[item['id']] for item in data[1]])

        preds = np.clip(np.exp(preds * 0.2 + 3.7), 0, 100)
        full_r = pearsonr(preds, targets)
        full_mae = mean_absolute_error(targets, preds)
        melters_r = pearsonr(preds[targets < 99], targets[targets < 99])
        melters_mae = mean_absolute_error(targets[targets < 99], preds[targets < 99])
        melters_f1 = f1_score(targets >= 99, preds >= 99)

    return full_r[0], full_mae, melters_r[0], melters_mae, melters_f1


TAPE_DIR = "/home/sheijl/tape-rq/tape"
if not os.path.exists(TAPE_DIR):
    TAPE_DIR = "/home/sheijl/tape"
MODELS_DIR = f"{TAPE_DIR}/melting_point_regr_mmseq_2"
TAPE_EVAL = "tape-eval"
if not os.path.exists(TAPE_EVAL):
    TAPE_EVAL = "tape-eval"
models = pandas.read_csv("/home/sheijl/Documents/models_chiba.csv").dropna()


def main(split="test"):
    real_valid_values = load_real_values(os.path.join(TAPE_DIR, "data/melting_temps/%s.lmdb" % split))

    # Generate varibench test results
    command = "cd {tape_dir}; {tape_eval} {model_type} melting_point_{prediction_type} {models_dir}/{run_name} " \
              "--tokenizer {tokenizer} --split %s --batch_size 1 --num_workers 0 --no_cuda" % split

    for r, row in models.iterrows():
        if not os.path.exists(os.path.join(TAPE_DIR, MODELS_DIR, row.run_name, "pytorch_model.bin")):
            print(f"Run name {row.run_name} not available.")
            continue

        if "unirep" in row.model_type.lower():
            tokenizer = "unirep"
        else:
            tokenizer = "iupac"

        cmd = command.format(
            tape_dir=TAPE_DIR,
            tape_eval=TAPE_EVAL,
            prediction_type=row.prediction_type,
            models_dir=MODELS_DIR,
            run_name=row.run_name,
            model_type=row.model_type.split("_")[0].lower(),
            tokenizer=tokenizer
        )
        print(cmd)

        try:
            results = get_stats(os.path.join(MODELS_DIR, row.run_name, "results.pkl"), real_valid_values)
        except (KeyError, FileNotFoundError):
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            print(process.communicate())
            results = get_stats(os.path.join(MODELS_DIR, row.run_name, "results.pkl"), real_valid_values)

        results_dict = {
            "full_r": round(results[0], 2),
            "full_mae": round(results[1], 2),
            "melters_r": round(results[2], 2),
            "melters_mae": round(results[3], 2),
            "melters_f1": round(results[4], 2)
        }
        out = row.to_dict()
        out.update(results_dict)
        del out["run_name"]

        with open("results_test.jsonl", "a+") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()