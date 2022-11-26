import os
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

try:
    from .download import download_model
except ImportError:
    from download import download_model

# Change this to your own path
PATH = os.getcwd()


def chatbot_predict(
    tokenizer: AutoTokenizer,
    model: AutoModelWithLMHead,
    dialouge_lenght: int,
    user_input: str,
    max_length: int = 500,
    top_k: int = 100,
    top_p: float = 0.7,
    temperature: float = 0.8,
) -> str:
    """
    Generate a dialogue with Socrates.

    Args:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelWithLMHead): The model.
        dialouge_lenght (int): Number of utterances in the dialogue.
        max_length (int): Maximum length of the dialogue.
        top_k (int): Number of tokens to sample from.
        top_p (float): Probability of sampling from the top_k tokens.
        temperature (float): Temperature of the sampling.

        Returns:
            str: The generated dialogue.
    """
    user_input = user_input

    for step in range(dialouge_lenght):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, return_tensors="pt"
        )

        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            if step > 0
            else new_user_input_ids
        )

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # return the bot response
        return "Socrates: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )


if __name__ == "__main__":
    if not os.listdir(os.path.join(PATH, "models/small-dialouGPT")):
        download_model(os.path.join(PATH, "models/small-dialouGPT"), "chatbot")

    os.chdir(PATH + "/models/small-dialouGPT/")
    tokenizer = AutoTokenizer.from_pretrained("./")
    model = AutoModelWithLMHead.from_pretrained("./")
    print(chatbot_predict(tokenizer, model, 5, "You are a philosopher."))
