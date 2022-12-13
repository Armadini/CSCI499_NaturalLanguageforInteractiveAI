# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch.nn.functional as F
import torch



class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, instructions_joined):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.instructions_joined = instructions_joined

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # # The linear layer that maps from hidden state space to action space
        # self.hidden2action = nn.Linear(hidden_dim, actionset_size)
        # # The linear layer that maps from hidden state space to object space
        # self.hidden2object = nn.Linear(hidden_dim, objectset_size)

    def forward(self, instructions):
        if self.instructions_joined:
            return self.forward_instructions_joined(instructions)
        else:
            return self.forward_instructions_seperate(instructions)

    def forward_instructions_joined(self, instructions):
        embeds = self.word_embeddings(instructions)
        _, final_hidden = self.lstm(embeds)
        # lstm_out = lstm_out[:, -1, :]
        return final_hidden
        # Actions
        # action_space = self.hidden2action(lstm_out)
        # action_scores = F.log_softmax(action_space, dim=1)
        # # Objects
        # object_space = self.hidden2object(lstm_out)
        # object_scores = F.log_softmax(object_space, dim=1)
        # return action_scores, object_scores

    def forward_instructions_seperate(self, instructions):
        pass


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_dim, hidden_dim, actionset_size, objectset_size):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to action space
        self.hidden2action_object = nn.Linear(hidden_dim, actionset_size+objectset_size)
        # The linear layer that maps from hidden state space to object space
        # self.hidden2object = nn.Linear(hidden_dim, objectset_size)

    def forward(self, x, hidden_state):

        lstm_out, final_hidden = self.lstm(x, hidden_state)
        # print("FOR THE LOVE OF GOD AND ALL THINGS HOLY")
        lstm_out = lstm_out[:, -1, :]

        action_object_scores = self.hidden2action_object(lstm_out)

        return final_hidden, action_object_scores # , action_scores, object_scores


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, actionset_size, objectset_size, instructions_joined, max_t, attention=False):
        super(EncoderDecoder, self).__init__()
        self.attention = attention
        self.actionset_size = actionset_size
        self.objectset_size = objectset_size
        self.max_t = max_t
        self.encoder = Encoder(embedding_dim, hidden_dim, vocab_size, instructions_joined)
        self.decoder = Decoder(actionset_size+objectset_size, hidden_dim, actionset_size, objectset_size)

    def forward(self, instructions, labels):
        final_hidden = self.encoder(instructions)

        action_preds = []
        object_preds = []
        for i in range(0, self.max_t, 1):
            teacher_labels = labels[:, :i, :] if i>0 else torch.zeros((labels.size(0), 1, labels.size(2)))
            final_hidden, a_o = self.decoder(teacher_labels.to(torch.float), final_hidden)
            action_preds.append(a_o[:, :self.actionset_size+1])
            object_preds.append(a_o[:, self.actionset_size+1:])
        return torch.stack(action_preds, 1), torch.stack(object_preds, 1)

    # TODO: ADD GREEDY DECODING
    # def forward_greedy(self, instructions):
    #     final_hidden = self.encoder(instructions)

    #     action_preds = []
    #     object_preds = []
    #     for i in range(0, self.max_t, 1):
    #         student_labels = labels[:, :i, :] if i>0 else torch.zeros((labels.size(0), 1, labels.size(2)))
    #         final_hidden, a_o = self.decoder(teacher_labels.to(torch.float), final_hidden)
    #         action_preds.append(a_o[:, :self.actionset_size+1])
    #         object_preds.append(a_o[:, self.actionset_size+1:])
    #     return torch.stack(action_preds, 1), torch.stack(object_preds, 1)
            
