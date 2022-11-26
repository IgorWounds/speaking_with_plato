import os
import logging
import tkinter as tk

from transformers import AutoModelWithLMHead, AutoTokenizer

from models.predict_chatbot_model import chatbot_predict
from models.predict_writebot_model import writebot_predict
from models.download import download_model


PATH = os.getcwd() + "/models/"
CHATBOT_PATH = os.getcwd() + "/models/small-dialouGPT/"
WRITEBOT_PATH = os.getcwd() + "/models/GPT-2/"

# Chatbot Parameters
CHATBOT_MAX_LENGTH = 500
CHATBOT_TOP_K = 100
CHATBOT_TOP_P = 0.7
CHATBOT_TEMPERATURE = 0.8

# Writebot Parameters
WRITEBOT_MAX_LENGTH = 250
WRITEBOT_TEMPERATURE = 0.6
WRITEBOT_TOP_K = 100
WRITEBOT_TOP_P = 0.9
WRITEBOT_N_SAMPLES = 5
WRITEBOT_BATCH_SIZE = 5

try:
    os.mkdir(CHATBOT_PATH)
except FileExistsError:
    pass

try:
    os.mkdir(WRITEBOT_PATH)
except FileExistsError:
    pass

# check if the models are downloaded
if not os.path.exists(CHATBOT_PATH) or len(os.listdir(WRITEBOT_PATH)) < 2:
    logging.info("Downloading the chatbot model...")
    download_model(CHATBOT_PATH, "chatbot")
    # make sure the model is unzipped
    if len(os.listdir(CHATBOT_PATH)) < 2:
        os.chdir(CHATBOT_PATH)
        os.system("tar -xvzf GPT-2.zip")


if not os.path.exists(WRITEBOT_PATH) or len(os.listdir(WRITEBOT_PATH)) < 2:
    logging.info("Downloading the writebot model...")
    download_model(WRITEBOT_PATH, "writebot")
    # make sure the model is unzipped
    if len(os.listdir(WRITEBOT_PATH)) < 2:
        os.chdir(WRITEBOT_PATH)
        os.system("tar -xvzf small-dialouGPT.zip")


os.chdir(PATH)


def start_chatbot():
    # create a frame for the chatbot
    frame = tk.Frame(root)
    frame.pack()

    # create a label for the chatbot
    label = tk.Label(frame, text="Speak with Socrates")
    label.config(font=("Courier", 30))
    label.pack()

    text_box = tk.Text(frame, height=30, width=80)
    text_box.config(font=("Courier", 20))
    text_box.pack()

    tokenizer = AutoTokenizer.from_pretrained(CHATBOT_PATH)
    model = AutoModelWithLMHead.from_pretrained(CHATBOT_PATH)

    def get_user_input():
        # get the user input
        user_input = text_box.get("1.0", "end-1c")
        return chatbot_predict(
            tokenizer=tokenizer,
            model=model,
            dialouge_lenght=1,
            user_input=user_input,
            max_length=CHATBOT_MAX_LENGTH,
            top_k=CHATBOT_TOP_K,
            top_p=CHATBOT_TOP_P,
            temperature=CHATBOT_TEMPERATURE,
        )

    def display_bot_response():
        bot_response = get_user_input()
        # display the user input and the bot response
        text_box.insert(tk.END, f"{bot_response} \n\n")

    def exit_chatbot():
        # check if the user is inside the chatbot
        if frame.winfo_exists():
            frame.destroy()
            start_menu(root)

    root.bind("<Return>", lambda event: display_bot_response())
    root.bind("<Escape>", lambda event: exit_chatbot())


def start_writebot():
    os.chdir(PATH + "/GPT-2/")
    # create a frame for the writebot
    frame = tk.Frame(root)
    frame.pack()

    # create a label for the writebot
    label = tk.Label(frame, text="Write with Plato")
    label.config(font=("Courier", 30))
    label.pack()

    text_box = tk.Text(frame, height=30, width=80)
    text_box.config(font=("Courier", 20))
    text_box.pack()

    def get_user_input():
        # wait for the user to write something
        user_input = text_box.get("1.0", "end-1c")

        if user_input:
            return str(
                writebot_predict(
                    prefix=str(user_input),
                    length=WRITEBOT_MAX_LENGTH,
                    temperature=WRITEBOT_TEMPERATURE,
                    top_k=WRITEBOT_TOP_K,
                    top_p=WRITEBOT_TOP_P,
                    nsamples=WRITEBOT_N_SAMPLES,
                    batch_size=WRITEBOT_BATCH_SIZE,
                )
            )

    def display_bot_response():
        bot_response = get_user_input()
        text_box.insert(tk.END, f"{bot_response} \n\n")

    def exit_writebot():
        if frame.winfo_exists():
            frame.destroy()
            start_menu(root)
            os.chdir(PATH)

    root.bind("<Return>", lambda event: display_bot_response())
    root.bind("<Escape>", lambda event: exit_writebot())


def buttons_callback(button_1, button_2, button_3, button_4, label, index):
    label.destroy()
    button_1.destroy()
    button_2.destroy()
    button_3.destroy()
    button_4.destroy()
    if index == 0:
        start_chatbot()
    elif index == 1:
        start_parameters("chatbot")
    elif index == 2:
        start_parameters("writebot")
    elif index == 3:
        start_writebot()


def start_parameters(model: str):
    frame = tk.Frame(root)
    frame.pack()

    # create a label for the chatbot
    if model == "chatbot":
        label = tk.Label(frame, text="Change Chatbot Parameters")
    else:
        label = tk.Label(frame, text="Change Writebot Parameters")

    label.config(font=("Courier", 30))
    label.pack()

    MAX_LENGTH_label = tk.Label(frame, text="Max Length:")
    MAX_LENGTH_label.config(font=("Courier", 20))
    MAX_LENGTH_label.pack()
    # create input boxes for the chatbot parameters and name them
    MAX_LENGTH_input = tk.Entry(frame, width=10)
    MAX_LENGTH_input.config(font=("Courier", 20))
    MAX_LENGTH_input.pack()

    MAX_LENGTH_label = tk.Label(frame, text="Top K:")
    MAX_LENGTH_label.config(font=("Courier", 20))
    MAX_LENGTH_label.pack()
    TOP_K_input = tk.Entry(frame, width=10)
    TOP_K_input.config(font=("Courier", 20))
    TOP_K_input.pack()

    MAX_LENGTH_label = tk.Label(frame, text="Top P:")
    MAX_LENGTH_label.config(font=("Courier", 20))
    MAX_LENGTH_label.pack()
    TOP_P_input = tk.Entry(frame, width=10)
    TOP_P_input.config(font=("Courier", 20))
    TOP_P_input.pack()

    MAX_LENGTH_label = tk.Label(frame, text="Temperature:")
    MAX_LENGTH_label.config(font=("Courier", 20))
    MAX_LENGTH_label.pack()
    TEMPERATURE_input = tk.Entry(frame, width=10)
    TEMPERATURE_input.config(font=("Courier", 20))
    TEMPERATURE_input.pack()

    if model == "writebot":
        WRITEBOT_N_SAMPLES_label = tk.Label(frame, text="N Samples:")
        WRITEBOT_N_SAMPLES_label.config(font=("Courier", 20))
        WRITEBOT_N_SAMPLES_label.pack()
        WRITEBOT_N_SAMPLES_input = tk.Entry(frame, width=10)
        WRITEBOT_N_SAMPLES_input.config(font=("Courier", 20))
        WRITEBOT_N_SAMPLES_input.pack()

        WRITEBOT_BATCH_SIZE_label = tk.Label(frame, text="Batch Size:")
        WRITEBOT_BATCH_SIZE_label.config(font=("Courier", 20))
        WRITEBOT_BATCH_SIZE_label.pack()
        WRITEBOT_BATCH_SIZE_input = tk.Entry(frame, width=10)
        WRITEBOT_BATCH_SIZE_input.config(font=("Courier", 20))
        WRITEBOT_BATCH_SIZE_input.pack()

        WRITEBOT_BATCH_SIZE_input.insert(0, WRITEBOT_BATCH_SIZE)
        WRITEBOT_N_SAMPLES_input.insert(0, WRITEBOT_N_SAMPLES)
        MAX_LENGTH_input.insert(0, WRITEBOT_MAX_LENGTH)
        TOP_K_input.insert(0, WRITEBOT_TOP_K)
        TOP_P_input.insert(0, WRITEBOT_TOP_P)
        TEMPERATURE_input.insert(0, WRITEBOT_TEMPERATURE)
    else:
        MAX_LENGTH_input.insert(0, CHATBOT_MAX_LENGTH)
        TOP_K_input.insert(0, CHATBOT_TOP_K)
        TOP_P_input.insert(0, CHATBOT_TOP_P)
        TEMPERATURE_input.insert(0, CHATBOT_TEMPERATURE)

    def exit_parameters():
        # check if the user is inside the chatbot
        if frame.winfo_exists():
            frame.destroy()
            start_menu(root)

    def change_parameters():
        global CHATBOT_MAX_LENGTH, CHATBOT_TOP_K, CHATBOT_TOP_P, CHATBOT_TEMPERATURE, WRITEBOT_MAX_LENGTH, WRITEBOT_TOP_K, WRITEBOT_TOP_P, WRITEBOT_TEMPERATURE, WRITEBOT_N_SAMPLES, WRITEBOT_BATCH_SIZE
        if model == "chatbot":
            CHATBOT_MAX_LENGTH = int(MAX_LENGTH_input.get())
            CHATBOT_TOP_K = int(TOP_K_input.get())
            CHATBOT_TOP_P = float(TOP_P_input.get())
            CHATBOT_TEMPERATURE = float(TEMPERATURE_input.get())
        else:
            WRITEBOT_MAX_LENGTH = int(MAX_LENGTH_input.get())
            WRITEBOT_TOP_K = int(TOP_K_input.get())
            WRITEBOT_TOP_P = float(TOP_P_input.get())
            WRITEBOT_TEMPERATURE = float(TEMPERATURE_input.get())
            WRITEBOT_N_SAMPLES = int(WRITEBOT_N_SAMPLES_input.get())
            WRITEBOT_BATCH_SIZE = int(WRITEBOT_BATCH_SIZE_input.get())

    # create a save button
    save_button = tk.Button(
        frame,
        text="Save",
        command=change_parameters,
        width=10,
        height=1,
        bg="green",
        fg="white",
    )
    save_button.config(font=("Courier", 20))
    save_button.pack(pady=30)

    root.bind("<Escape>", lambda event: exit_parameters())


def start_menu(root):
    menu = tk.Menu(root)
    root.config(menu=menu)
    root.title("Speaking with Plato")
    root.geometry("930x750")
    root.config(bg="white")
    root.iconbitmap("../src/Platon.ico")

    # create a label to go on the main page
    label = tk.Label(root, text="Welcome to the Speaking with Plato App")
    label.config(font=("Courier", 30))
    label.pack(pady=15)

    # create a button to start the chatbot in a new window
    chatbot_button = tk.Button(
        root,
        text="Start Chatbot",
        command=lambda: buttons_callback(
            chatbot_button,
            chatbot_edit_button,
            writebot_button,
            writebot_edit_button,
            label,
            0,
        ),
    )
    chatbot_button.config(
        height=6, width=20, font=("Courier", 14), bg="black", fg="white"
    )
    chatbot_button.pack(side=tk.TOP, pady=15)
    chatbot_button.pack()

    chatbot_edit_button = tk.Button(
        root,
        text="Edit Chatbot",
        command=lambda: buttons_callback(
            writebot_button,
            chatbot_edit_button,
            chatbot_button,
            writebot_edit_button,
            label,
            1,
        ),
    )
    chatbot_edit_button.config(
        height=6, width=20, font=("Courier", 14), bg="black", fg="white"
    )
    chatbot_edit_button.pack(side=tk.TOP, pady=15)
    chatbot_edit_button.pack()

    writebot_button = tk.Button(
        root,
        text="Start Writebot",
        command=lambda: buttons_callback(
            writebot_button,
            chatbot_edit_button,
            chatbot_button,
            writebot_edit_button,
            label,
            3,
        ),
    )
    writebot_button.config(
        height=6, width=20, font=("Courier", 14), bg="black", fg="white"
    )
    writebot_button.pack(side=tk.TOP, pady=15)
    writebot_button.pack()

    writebot_edit_button = tk.Button(
        root,
        text="Edit Writebot",
        command=lambda: buttons_callback(
            writebot_button,
            chatbot_edit_button,
            chatbot_button,
            writebot_edit_button,
            label,
            2,
        ),
    )
    writebot_edit_button.config(
        height=6, width=20, font=("Courier", 14), bg="black", fg="white"
    )
    writebot_edit_button.pack(side=tk.TOP, pady=15)
    writebot_edit_button.pack()

    root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    start_menu(root)
