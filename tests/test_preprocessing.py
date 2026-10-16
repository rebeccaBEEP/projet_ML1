import pandas as pd

from winequality.preprocessing import FEATURES, load_data, make_target, preprocess


def test_load_data_concatene_rouge_et_blanc(data_dir):
    df = load_data(data_dir)
    assert len(df) == 700
    assert set(df["type"]) == {"red", "white"}


def test_make_target_seuil_7():
    df = pd.DataFrame({"quality": [3, 6, 7, 9]})
    assert make_target(df)["bon"].tolist() == [0, 0, 1, 1]


def test_preprocess_supprime_les_doublons(data_dir):
    df = load_data(data_dir)
    df_dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
    X, y = preprocess(df_dup)
    assert len(X) == len(df)
    assert len(X) == len(y)


def test_preprocess_encode_type_et_ordre_des_colonnes(data_dir):
    X, y = preprocess(load_data(data_dir))
    assert list(X.columns) == FEATURES
    assert set(X["type"]) == {0, 1}
    assert set(y) <= {0, 1}
    assert not X.isnull().any().any()
