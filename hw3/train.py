import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)

import lang_to_sem_loader
from model import EncoderDecoder


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    return lang_to_sem_loader.get_loaders(input_path=args.in_data_fn, batch_size=7, shuffle=True, debug=args.debug, join_instructions=args.join_instructions)


def setup_model(args, metadata):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    model = EncoderDecoder(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, vocab_size=args.vocab_size, actionset_size=metadata["max_actions"], objectset_size=metadata["max_objects"], instructions_joined=args.join_instructions, max_t=metadata["seq_len"], attention=args.attention)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion, target_criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        action_logits, target_logits = model(inputs, labels)

        action_loss = action_criterion(
            action_logits.squeeze(), labels[:, :, :model.actionset_size+1].float())
        target_loss = target_criterion(
            target_logits.squeeze(), labels[:, :, model.actionset_size+1:].float())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        # gets top k predictions in format (values, indices)
        # we care about indices bc of one hot encoding
        action_vals, action_preds = action_logits.topk(1, dim=2)
        target_vals, target_preds = target_logits.topk(1, dim=2)
        action_label_vals, action_labels = labels[:,:,:10].topk(1, dim=2)
        target_label_vals, target_labels = labels[:,:,10:].topk(1, dim=2)

        action_preds, action_labels = torch.flatten(action_preds, start_dim=1), torch.flatten(action_labels, start_dim=1)
        target_preds, target_labels = torch.flatten(target_preds, start_dim=1), torch.flatten(target_labels, start_dim=1)

        # print("ACTION PREDICTIONS SHAPE", action_preds.size())
        # print("ACTION LABELS SHAPE", action_labels.size())
        # print("TARGET PREDICTIONS SHAPE", target_preds.size())
        # print("TARGET LABELS SHAPE", target_labels.size())

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        action_em = action_preds == action_labels
        target_em = target_preds == target_labels
        em = action_em & target_em
        # print("ACTION EM", action_em)
        # print("TARGET EM", target_em)
        # print("EM", em)
        # em = output == labels[:, 0]
        # prefix_em = prefix_em(output, labels)
        acc = accuracy_score(action_labels.flatten(), action_preds.flatten()) + accuracy_score(target_labels.flatten(), target_preds.flatten())

        # logging
        epoch_loss += loss.item()
        epoch_acc += acc

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, action_criterion, target_criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion, target_criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(int(args.num_epochs))):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion, target_criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % float(args.val_every) == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, metadata = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    # , maps, device, 
    model = setup_model(args, metadata)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, action_criterion, target_criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument('--join_instructions', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="size of the embedding of each word in the model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of the hidden state produced/consumed by the LSTM"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=1000, help="number of tokens in our vocabulary (including pad, start, end, unk)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-1, help="the learning rate for the optimizer"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()
    args.debug = False
    print(f"DEBUG {args.debug}")

    main(args)
