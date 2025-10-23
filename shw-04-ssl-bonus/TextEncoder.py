import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        """
        Create the model and set its weights frozen. 
        Use Transformers library docs to find out how to do this.
        """
        # use the CLS token hidden representation as the sentence's embedding
        if pretrained:
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        for param in self.model.parameters():
            param.requires_grad = trainable
        
        self.output_dim = self.model.config.hidden_size
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Pass the arguments through the model and make sure to return CLS token embedding
        """
        # Пропускаем входные данные через модель
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True  # Возвращаем выход в виде словаря
        )
        
        # Извлекаем последние скрытые состояния
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # Берем скрытое состояние CLS токена (первый токен в последовательности)
        cls_embedding = last_hidden_state[:, self.target_token_idx, :]  # [batch_size, hidden_dim]
        
        return cls_embedding
