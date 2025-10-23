import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.drop = nn.Dropout(0.3)
        self.RNN = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.Embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, padding_idx=self.dataset.pad_id)


    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """

        embed = self.drop(self.Embed(indices))
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.RNN(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=indices.size(1))
        logits = self.fc(out)

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str='', max_len: int = None, temp:float = 1.0, top_k=10) -> str:
        self.eval()
        if max_len is None:
            max_len = self.max_length

        prefix_ids = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        input_ids = torch.tensor([prefix_ids], device=next(self.parameters()).device)
        embed = self.Embed(input_ids)
        out, hidden = self.RNN(embed)

        generated = prefix_ids[:]
        for _ in range(max_len - len(prefix_ids)):
            last_token = torch.tensor([[generated[-1]]], device=input_ids.device)
            embed = self.Embed(last_token)
            out, hidden = self.RNN(embed, hidden)
            logits = self.fc(out[:, -1, :]) / temp

                # топ-k сэмплинг
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, top_k)
            next_token = top_idx[0, torch.multinomial(top_probs / top_probs.sum(), 1)].item()


            if next_token == self.dataset.eos_id:
                break
            generated.append(next_token)

        return self.dataset.ids2text(generated[1:])


