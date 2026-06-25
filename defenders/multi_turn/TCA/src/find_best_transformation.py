import numpy as np
import pandas as pd
from defenders.multi_turn.TCA.src.utility import transform_feature
from sklearn.feature_selection import mutual_info_classif



if __name__ == "__main__":
    
    df= pd.read_csv("defenders/multi_turn/integrated/data/total/features_before_selection(7).csv")

    target_column = "label"

    feature_columns = [
        col
        for col in df.columns
        if col not in ["label", "conv_id", "turn_id"]
    ]
    results = []

    transformations = [
        "original",
        "log1p",
        "square",
        "binarize",
        "yeo-johnson"
    ]

    for feature in feature_columns:


        best_score = -1
        best_transform = None

        for transform in transformations:

            try:

                transformed = transform_feature(
                    df[feature],
                    transform
                )

                X = np.array(transformed).reshape(-1, 1)
                #select depending on the mutual info gained 
                score = mutual_info_classif(
                    X,
                    df[target_column],
                    random_state=42
                )[0]

                results.append({
                    "feature": feature,
                    "transform": transform,
                    "score": score
                })

                if score > best_score:
                    best_score = score
                    best_transform = transform

            except Exception as e:

                print(
                    f"Failed {feature} with {transform}: {e}"
                )
                pass
            
            
    results_df = pd.DataFrame(results)
    best_transforms = (
        results_df
        .sort_values("score", ascending=False)
        .groupby("feature")
        .first()
        .reset_index()
    )

    result = {
        row["feature"]: row["transform"]
        for _, row in best_transforms.iterrows()
    }

    py_file_path = "defenders/multi_turn/TCA/inference/transforms.py"

    with open(py_file_path, "w") as f:
        f.write("TRANSFORMS = ")
        f.write(repr(result))

    print("Saved:", py_file_path)