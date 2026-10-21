"""Step 1: match CSV rows to image files and create a leak-free train/val/test split.

Split is done by PATIENT (not by image): one patient has several near-identical photos,
so splitting by image would leak the same patient into train and test and give fake accuracy.
"""
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import config as C


def index_folder(folder):
    """{lowercase file stem: path} for every image in a folder."""
    if not folder.is_dir():
        return {}
    return {p.stem.lower(): p for p in folder.iterdir() if p.suffix.lower() in C.IMAGE_EXTS}


def main():
    df = pd.read_csv(C.METADATA_CSV)
    df.columns = [c.strip() for c in df.columns]
    print(f"Metadata rows: {len(df)}")

    # --- match each row to a file ---------------------------------------------------
    paths, cache = [], {}
    for _, r in df.iterrows():
        folder = C.IMAGE_DIR / r["Folder_name"]
        if folder not in cache:
            cache[folder] = index_folder(folder)
        p = cache[folder].get(str(r["Image_id"]).strip().lower())
        paths.append(str(p) if p else None)
    df["path"] = paths

    missing = df["path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} CSV rows have no matching image file. Examples:")
        print(df[df.path.isna()][["Folder_name", "Image_id"]].head(10).to_string())
        if missing == len(df):
            sys.exit("No files matched. Check DATA_DIR in config.py and the folder layout.")
    df = df.dropna(subset=["path"]).copy()
    df["label"] = df["Diagnosis"].str.strip()

    # --- patient-level stratified split: ~80 / 10 / 10 -------------------------------
    s1 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=C.SEED)
    trainval_idx, test_idx = next(s1.split(df, df["label"], df["Patient_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    s2 = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=C.SEED)
    tr_idx, val_idx = next(s2.split(trainval, trainval["label"], trainval["Patient_id"]))
    train, val = trainval.iloc[tr_idx], trainval.iloc[val_idx]

    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    # --- safety check: no patient appears in two splits ------------------------------
    sets = {k: set(g["Patient_id"]) for k, g in df.groupby("split")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), \
        "Patient leakage between splits!"

    C.OUTPUT_DIR.mkdir(exist_ok=True)
    df[["path", "label", "Patient_id", "split"]].to_csv(C.SPLITS_CSV, index=False)

    print("\nImages per class per split:")
    print(pd.crosstab(df["label"], df["split"])[["train", "val", "test"]])
    print(f"\nSaved -> {C.SPLITS_CSV}")


if __name__ == "__main__":
    main()
