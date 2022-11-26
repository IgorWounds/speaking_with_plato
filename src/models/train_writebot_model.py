import os
import gpt_2_simple as gpt2

PATH = "C:/Users/igorr/Documents/GitHub/speaking_plato/speaking_with_plato"

# check if path is empty
def download_model(model_name: str = "GPT-2", model_type: str = "355M"):
    if not os.listdir(PATH + "/models/" + model_name):
        gpt2.download_gpt2(model_name=model_type)


os.chdir(PATH + "/models/GPT-2/")


def train(
    model_name: str = "355M",
    steps: int = 1000,
    print_every: int = 100,
    sample_every: int = 200,
    save_every: int = 500,
) -> None:
    """Train a GPT-2 model on all Plato's works.

    Args:
        model_name (str, optional): Name of the GPT-2 model. Defaults to "355M".
        steps (int, optional): Number of training steps. Defaults to 1000.
        print_every (int, optional): Print prediction every number of steps. Defaults to 100.
        sample_every (int, optional): Sample every number of steps. Defaults to 200.
        save_every (int, optional): Save checkpoint every number of steps. Defaults to 500.
    """
    download_model()
    sess = gpt2.start_tf_sess()
    gpt2.finetune(
        sess,
        dataset=PATH + "/data/processed/all_plato.txt",
        model_name=model_name,
        steps=steps,
        restore_from="fresh",
        run_name="run_1",
        print_every=print_every,
        sample_every=sample_every,
        save_every=save_every,
    )


if __name__ == "__main__":
    train()
