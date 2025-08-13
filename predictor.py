import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, SiglipModel
from PIL import Image
import re

# Configuration
class Config:
    image_size = 224
    embed_dim = 512
    temperature = 0.07
    dropout_rate = 0.1

def filter_single_characters(text):
    """Enhanced text filtering for both English and Arabic"""
    if not isinstance(text, str):
        text = str(text)
    
    words = text.split()
    filtered_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        
        if len(clean_word) == 1:
            is_arabic = bool(re.match(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', clean_word))
            is_alpha = clean_word.isalpha()
            
            if is_arabic or is_alpha:
                continue
        
        if len(clean_word) > 1 or (len(clean_word) == 1 and not clean_word.isalpha()):
            filtered_words.append(word)
    
    filtered_text = ' '.join(filtered_words).strip()
    return filtered_text if filtered_text else text

class EnhancedSigLIP(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.model = SiglipModel.from_pretrained(model_name)
        self.temperature = nn.Parameter(torch.tensor(Config.temperature))
        
        # Enhanced projection heads
        self.text_proj = nn.Sequential(
            nn.Linear(self.model.config.text_config.hidden_size, Config.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.embed_dim * 2, Config.embed_dim)
        )
        
        self.vision_proj = nn.Sequential(
            nn.Linear(self.model.config.vision_config.hidden_size, Config.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.embed_dim * 2, Config.embed_dim)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        text_embeds = F.normalize(self.text_proj(outputs.text_embeds), p=2, dim=-1)
        image_embeds = F.normalize(self.vision_proj(outputs.image_embeds), p=2, dim=-1)
        
        return text_embeds, image_embeds

class SimilarityPredictor:
    def __init__(self, model_path, threshold=0.5):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model (.pth file)
            threshold: Similarity threshold for classification (default: 0.5)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        print(f"Using device: {self.device}")
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        
        print("Loading model...")
        self.model = EnhancedSigLIP().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.model.eval()
    
    def predict_similarity(self, image_path, text, verbose=True):
        """
        Predict if image and text are similar
        
        Args:
            image_path: Path to the image file
            text: Text description
            verbose: Whether to print detailed results
        
        Returns:
            dict: Contains similarity score, prediction, and confidence
        """
        try:
            # Load and process image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # Process text
            original_text = str(text).strip()
            filtered_text = filter_single_characters(original_text)
            
            if verbose:
                print(f"📝 Original text: '{original_text}'")
                if original_text != filtered_text:
                    print(f"🔍 Filtered text: '{filtered_text}'")
                print(f"🖼️  Image: {image_path}")
            
            # Process inputs
            inputs = self.processor(
                text=filtered_text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            pixel_values = inputs['pixel_values'].to(self.device)
            
            if 'attention_mask' in inputs:
                attention_mask = inputs['attention_mask'].to(self.device)
            else:
                attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long()
            
            # Get predictions
            with torch.no_grad():
                text_embeds, image_embeds = self.model(input_ids, attention_mask, pixel_values)
                
                # Calculate similarity
                similarity = torch.dot(text_embeds[0], image_embeds[0]).item()
                
                # Make prediction
                prediction = similarity > self.threshold
                confidence = abs(similarity - self.threshold)
                
                result = {
                    'similarity_score': similarity,
                    'prediction': 'MATCH' if prediction else 'NO MATCH',
                    'is_match': prediction,
                    'confidence': confidence,
                    'threshold': self.threshold
                }
                
                if verbose:
                    print(f"\n🎯 Results:")
                    print(f"   Similarity Score: {similarity:.4f}")
                    print(f"   Threshold: {self.threshold}")
                    print(f"   Prediction: {result['prediction']}")
                    print(f"   Confidence: {confidence:.4f}")
                    
                    if prediction:
                        print("✅ The image and text are SIMILAR!")
                    else:
                        print("❌ The image and text are NOT similar.")
                
                return result
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None

def quick_test(model_path, image_path, text, threshold=0.5):
    """
    Quick function to test a single image-text pair
    
    Args:
        model_path: Path to your trained model
        image_path: Path to the image
        text: Text description
        threshold: Similarity threshold (default: 0.5)
    """
    predictor = SimilarityPredictor(model_path, threshold)
    result = predictor.predict_similarity(image_path, text)
    return result