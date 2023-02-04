from transformers import AutoTokenizer, AutoConfig, AutoModel

import torch

class CustomModel(torch.nn.Module):
    """
    This takes a transformer backbone and puts a slightly-modified classification head on top.
    
    """

    def __init__(self, model_name, num_extra_dims, num_labels=2):
        # num_extra_dims corresponds to the number of extra dimensions of numerical/categorical data

        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        num_hidden_size = self.transformer.config.hidden_size # May be different depending on which model you use. Common sizes are 768 and 1024. Look in the config.json file 

        self.linear_layer_1 = torch.nn.Linear(num_hidden_size+num_extra_dims, 32)
        # Output size is 1 since this is a binary classification problem
        self.linear_layer_2 = torch.nn.Linear(32, 16)
        self.linear_layer_output = torch.nn.Linear(16, 1)
        self.relu = torch.nn.LeakyReLU(0.6)
        self.dropout_1 = torch.nn.Dropout(0.5)


    def forward(self, input_ids, extra_features, attention_mask=None, token_type_ids=None, labels=None):
        """
        extra_features should be of shape [batch_size, dim] 
        where dim is the number of additional numerical/categorical dimensions
        """

        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # [batch size, sequence length, hidden size]

        cls_embeds = hidden_states.last_hidden_state[:, 0, :] # [batch size, hidden size]

        concat = torch.cat((cls_embeds, extra_features), dim=-1) # [batch size, hidden size+num extra dims]

        output_1 = self.relu(self.linear_layer_1(concat)) # [batch size, num labels]
        output_2 = self.relu(self.linear_layer_2(output_1))
        final_output = self.dropout_1(self.linear_layer_output(output_2))

        return final_output