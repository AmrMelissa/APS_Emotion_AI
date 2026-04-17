import torch
import torch.nn as nn
from transformers import BertModel

class BertWithStatisticalPooling(nn.Module):
    """
    BERT + Statistical Pooling pour la classification d'émotions et la Feature Fusion.
    Ce fichier centralise l'architecture pour qu'elle puisse être importée par les scripts
    d'entraînement (train_bert_slurm) et par les scripts de fusion de Kawther.
    """
    
    def __init__(self, model_name, num_classes, dropout=0.1):
        super(BertWithStatisticalPooling, self).__init__()
        
        # Chargement du modèle de base
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Tête de classification
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert_hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """
        Paramètres:
        - input_ids: Tokens du texte
        - attention_mask: Masque d'attention pour ignorer le padding
        - return_embeddings: Si True, renvoie le vecteur (text_vec) sans passer par le classifieur final
                             Essentiel pour l'étape de Fusion "Feature Fusion" !
        """
        # 1. Extraction des features avec BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Mean Pooling avec masquage (ne pas moyenner le padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        # Éviter la division par zéro
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        # 3. Arrêt intermédiaire pour la Feature Fusion (Architecture 2)
        if return_embeddings:
            # On renvoie le text_vec (dimension d = 768 pour BERT de base)
            return pooled_output
            
        # 4. Classification classique
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
