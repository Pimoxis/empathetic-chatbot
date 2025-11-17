import torch


BASE_MODEL_NAME = "microsoft/DialoGPT-medium"

EMPATHETIC_MODEL_NAME = "empathetic-dialogpt-medium"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 128        
MAX_HISTORY_TURNS = 6       

TEMPERATURE = 0.7            
TOP_P = 0.9                  
REPETITION_PENALTY = 1.1     

EMOTION_DATASET_NAME = "facebook/empathetic_dialogues"
