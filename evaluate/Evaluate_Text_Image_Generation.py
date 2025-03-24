import torch
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import lpips
import numpy as np
from torchvision import transforms, models
from tqdm import tqdm
import os
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

class InceptionV3Features:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.to(device)
        self.model.eval()
        self.model.fc = torch.nn.Identity()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_features(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            features = self.model(img)
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

class TextImageEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize CLIP model from HuggingFace
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Initialize Inception model for FID
        self.inception = InceptionV3Features(device)
        
        # Standard image transform for LPIPS
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def compute_clip_image_similarity(self, image1_path, image2_path):
        """Compute CLIP-I similarity between two images"""
        try:
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')
            
            inputs = self.clip_processor(images=[image1, image2], return_tensors="pt", padding=True)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(inputs['pixel_values'])
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                similarity = torch.nn.functional.cosine_similarity(
                    image_features[0].unsqueeze(0),
                    image_features[1].unsqueeze(0)
                ).item()
            
            return similarity
        except Exception as e:
            print(f"Error computing CLIP-I similarity: {e}")
            return None

    def compute_clip_text_similarity(self, image_path, text_prompt):
        """Compute CLIP-T similarity between image and text"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.clip_processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            
            inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(inputs['pixel_values'])
                text_features = self.clip_model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                similarity = torch.nn.functional.cosine_similarity(
                    image_features,
                    text_features
                ).item()
            
            return similarity
        except Exception as e:
            print(f"Error computing CLIP-T similarity: {e}")
            return None

    def compute_lpips_distance(self, image1_path, image2_path):
        """Compute LPIPS perceptual distance between two images"""
        try:
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_distance = self.lpips_model(img1_tensor, img2_tensor).item()
            
            return lpips_distance
        except Exception as e:
            print(f"Error computing LPIPS distance: {e}")
            return None

    def compute_fid(self, real_features, fake_features):
        try:
            mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
            
            ssdiff = np.sum((mu1 - mu2) ** 2)
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            return float(fid)
        except Exception as e:
            print(f"Error computing FID score: {e}")
            return None

def evaluate_generations(json_path, output_base_path="", reference_base_path=""):
    """Evaluate text-to-image generations using multiple metrics"""
    evaluator = TextImageEvaluator()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    text_image_entries = [entry for entry in data if entry["category"] == "Text-Image_Generation"]
    os.makedirs("evaluation_results", exist_ok=True)
    
    real_features = []
    fake_features = []
    
    print("Extracting image features for FID calculation...")
    for entry in tqdm(text_image_entries, desc="Processing images"):
        try:
            output_path = os.path.join(output_base_path, entry["output"])
            reference_path = os.path.join(reference_base_path, entry["data"]["image"])
            
            if not os.path.exists(output_path) or not os.path.exists(reference_path):
                print(f"Skipping entry due to missing files:\nOutput: {output_path}\nReference: {reference_path}")
                continue
            
            real_feat = evaluator.inception.get_features(reference_path)
            fake_feat = evaluator.inception.get_features(output_path)
            
            if real_feat is not None and fake_feat is not None:
                real_features.append(real_feat)
                fake_features.append(fake_feat)
            
        except Exception as e:
            print(f"Error processing entry for FID: {e}")
            continue
    
    real_features = np.stack(real_features)
    fake_features = np.stack(fake_features)
    fid_score = evaluator.compute_fid(real_features, fake_features)
    
    print("Computing individual metrics...")
    for entry in tqdm(text_image_entries, desc="Evaluating generations"):
        try:
            output_path = os.path.join(output_base_path, entry["output"])
            reference_path = os.path.join(reference_base_path, entry["data"]["image"])
            text_prompt = entry["Text_Prompt"]
            
            if not os.path.exists(output_path) or not os.path.exists(reference_path):
                continue
            
            clip_i = evaluator.compute_clip_image_similarity(output_path, reference_path)
            clip_t = evaluator.compute_clip_text_similarity(output_path, text_prompt)
            lpips_d = evaluator.compute_lpips_distance(output_path, reference_path)
            
            result = {
                "output_path": entry["output"],
                "reference_path": entry["data"]["image"],
                "text_prompt": text_prompt,
                "metrics": {
                    "CLIP-I": clip_i,
                    "CLIP-T": clip_t,
                    "LPIPS-D": lpips_d
                }
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    
    avg_metrics = {
        "CLIP-I": np.mean([r["metrics"]["CLIP-I"] for r in results if r["metrics"]["CLIP-I"] is not None]),
        "CLIP-T": np.mean([r["metrics"]["CLIP-T"] for r in results if r["metrics"]["CLIP-T"] is not None]),
        "LPIPS-D": np.mean([r["metrics"]["LPIPS-D"] for r in results if r["metrics"]["LPIPS-D"] is not None]),
        "FID": fid_score
    }
    
    output_results = {
        "individual_results": results,
        "average_metrics": avg_metrics
    }
    
    results_path = os.path.join("evaluation_results", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    return output_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate text-to-image generations")
    parser.add_argument("--json_path", type=str, required=True, 
                       help="Path to the JSON file containing generation data")
    parser.add_argument("--output_base_path", type=str, default="", 
                       help="Base path for generated images (output)")
    parser.add_argument("--reference_base_path", type=str, default="", 
                       help="Base path for reference images (data/image)")
    
    args = parser.parse_args()
    
    results = evaluate_generations(
        args.json_path, 
        args.output_base_path, 
        args.reference_base_path
    )
    
    print("\nAverage Metrics:")
    for metric, value in results["average_metrics"].items():
        print(f"{metric}: {value:.4f}")