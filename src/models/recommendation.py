import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class DentalRecommendationSystem(nn.Module):
    def __init__(self, num_labels=3, model_name='bert-base-uncased'):
        super(DentalRecommendationSystem, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.recommendation_templates = {
            0: [  # Mild caries
                "Regular brushing twice daily with fluoride toothpaste is recommended.",
                "Consider using dental floss daily to prevent interdental caries.",
                "Schedule a follow-up appointment in 6 months."
            ],
            1: [  # Moderate caries
                "Immediate dental consultation is recommended for filling treatment.",
                "Use a high-fluoride toothpaste prescribed by your dentist.",
                "Reduce sugar intake and maintain strict oral hygiene."
            ],
            2: [  # Severe caries
                "Urgent dental treatment required to prevent further decay.",
                "Root canal treatment might be necessary.",
                "Consider crown placement after treatment."
            ]
        }
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def get_recommendations(self, severity_scores, patient_history=None):
        """
        Generate personalized recommendations based on severity scores and patient history
        
        Args:
            severity_scores (torch.Tensor): Model predictions for caries severity
            patient_history (dict, optional): Patient's dental history and habits
        
        Returns:
            list: Personalized recommendations
        """
        recommendations = []
        severity_level = torch.argmax(severity_scores).item()
        
        # Add severity-specific recommendations
        recommendations.extend(self.recommendation_templates[severity_level])
        
        # Add personalized recommendations based on patient history
        if patient_history:
            if patient_history.get('brushing_frequency', 0) < 2:
                recommendations.append(
                    "Increase brushing frequency to at least twice daily."
                )
            if not patient_history.get('uses_fluoride', True):
                recommendations.append(
                    "Switch to a fluoride-containing toothpaste for better protection."
                )
            if patient_history.get('sugar_consumption', 'low') == 'high':
                recommendations.append(
                    "Reduce consumption of sugary foods and drinks."
                )
        
        return recommendations
    
    def preprocess_input(self, text):
        """
        Preprocess input text for the model
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded
    
    @torch.no_grad()
    def predict(self, text):
        """
        Make predictions on input text
        """
        encoded = self.preprocess_input(text)
        outputs = self(encoded['input_ids'], encoded['attention_mask'])
        predictions = torch.softmax(outputs, dim=1)
        return predictions 