import os
from typing import Optional
import gpt_2_simple as gpt2

try:
    from .download import download_model
except ImportError:
    from download import download_model

# Change this to your own path
PATH = os.getcwd()


def writebot_predict(
    length: int = 250,
    temperature: float = 0.6,
    nsamples: int = 5,
    batch_size: int = 5,
    top_k: int = 100,
    top_p: float = 0.9,
    prefix: Optional[str] = None,
    checkpoint_dir: str = os.path.join(PATH, "models", "GPT-2", "checkpoint"),
) -> str:
    """Generate text using the GPT-2 model.

    Args:
        length (int): Number of tokens to generate.
        temperature (float): Temperature for sampling.
        nsamples (int): Number of samples to generate.
        batch_size (int): Number of batches to generate.
        top_k (int): Number of top tokens to sample from.
        top_p (float): Probability of sampling from top tokens.
        prefix (Optional[str]): Prefix to start the generation with.

    Returns:
        str: Generated text.
    """
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name="run_1", checkpoint_dir=checkpoint_dir)
    # obtain the generated text as a string

    # generate text into a text variable
    text = gpt2.generate(
        sess,
        length=length,
        temperature=temperature,
        nsamples=nsamples,
        batch_size=batch_size,
        top_k=top_k,
        top_p=top_p,
        prefix=prefix,
        include_prefix=True if prefix else False,
        run_name="run_1",
        checkpoint_dir=checkpoint_dir,
        return_as_list=True,
    )

    # convert the list of strings into a single string
    text = " \n================================\n".join(text)

    return text


if __name__ == "__main__":
    if not os.listdir(os.path.join(PATH, "models/GPT-2")):
        download_model(os.path.join(PATH, "models/GPT-2"), "writebot")

    os.chdir(PATH + "/models/GPT-2/")
    print(writebot_predict(prefix="You are a philosopher."))
