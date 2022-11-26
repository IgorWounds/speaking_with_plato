import pandas as pd
from sklearn.model_selection import train_test_split

# pat to file is the processed dialogue file
PATH_TO_FILE = "C:/Users/igorr/Documents/GitHub/speaking_plato/speaking_with_plato/data/processed/dialogues.txt"
CONTEXT_N = 15


def create_train_test(
    path_to_file: str = PATH_TO_FILE, context_n: int = CONTEXT_N
) -> pd.DataFrame:
    """
    Create dialogue features from a text file, split the data into train and test sets,
    and save them to a csv file.

    Args:
        path_to_file (str): Path to the text file containing the dialogue data.
        context_n (int): Number of previous utterances to include in the context.

    Returns:
        pd.DataFrame: Two dataframes, one for the train set and one for the test set.
    """
    # find the directory of the file

    dialogues = open(PATH_TO_FILE, "r").read()

    df = pd.DataFrame(dialogues.split("\n"), columns=["dialogue"])
    df["speaker"] = df["dialogue"].str.split(":").str[0]
    df["dialogue"] = df["dialogue"].str.split(":").str[1]

    contexted = []

    # Create contexted dialogues
    for i in range(context_n, len(df["dialogue"])):
        row = []
        prev = i - 1 - context_n
        for j in range(i, prev, -1):
            row.append(df["dialogue"][j])
        contexted.append(row)

    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(context_n - 1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)
    train_df, test_df = train_test_split(df, test_size=0.1)

    return train_df, test_df
