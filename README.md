# Assignment3
Running the Notebook

This notebook is structured in a way that all cells can be executed sequentially without modification. You may also use the Run All Cells option for convenience. However, please be cautious when executing WandB sweep cells at the end, as they may trigger multiple training runs.

Running the Model Without WandB
To evaluate a single model without using Weights & Biases (WandB), run the following command:


model = test_on_dataset(language="hi",
                        embedding_dim=256,
                        encoder_layers=3,
                        decoder_layers=3,
                        layer_type="lstm",
                        units=256,
                        dropout=0.2,
                        attention=False)

Running the Model With WandB Sweep
To perform a hyperparameter sweep using WandB, follow these steps:

1. Define the Sweep Configuration:

sweep_config = {
  "name": "Sweep 1- Assignment3",
  "method": "grid",
  "parameters": {
        "enc_dec_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [64, 128, 256]
        },
        "layer_type": {
            "values": ["rnn", "gru", "lstm"]
        } 
    }
}

2. Create the Sweep:

sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment3")
3. Launch the Sweep Agent:

wandb.agent(sweep_id, function=lambda: train_with_wandb("hi"))

Visualizing Model Outputs Using WordCloud
To visualize model outputs in a wordcloud format, use the following command (assumes model is already trained):

visualize_model_outputs(model, n=20)
This will generate a wordcloud of 20 samples, with words colored on a VIBGYOR scale to reflect BLEU score quality.

Visualizing Model Connectivity
To visualize internal model attention or activation connectivity for selected test words:

1. Sample Test Words:

test_words = get_test_words(5)

2. Visualize the Connectivity:

for word in test_words:
    visualise_connectivity(model, word, activation="scaler")
This helps analyze how the model attends to or transforms input sequences during inference.

3.To create the English to hindi Transliteration by model csv file use the following command (change attention to false for vanilla predictions)
model = test_on_dataset(language="hi",
                        embedding_dim=256,
                        encoder_layers=3,
                        decoder_layers=3,
                        layer_type="lstm",
                        units=256,
                        dropout=0.2,
                        attention=True,
                        save_outputs= "Eng_to_Hin_With_att.csv")

WANDB Report link: https://wandb.ai/ns24z347-indian-institute-of-technology-madras/Assignment3/reports/Roll-No-NS24Z347-Assignment-3--VmlldzoxMjg2NjIyNA/edit?draftId=VmlldzoxMjg2NjIyNA==
