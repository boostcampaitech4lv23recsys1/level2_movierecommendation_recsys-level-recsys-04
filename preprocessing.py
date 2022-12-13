import pandas as pd


def main():
    """
    genre에 label encoding 적용 후, item 단위 list로 묶은 후 json으로 저장
    ex) 
    {
        3: [4, 2, 5]
        11: [3, 7, 2, 2, 11],
    }
    """
    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "data/Ml_item2attributes.json"
    )


if __name__ == "__main__":
    main()
