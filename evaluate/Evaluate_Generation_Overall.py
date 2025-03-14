#!/usr/bin/env python3
import json
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from torch.nn.functional import mse_loss, interpolate
from scipy import linalg
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
from tqdm import tqdm
import lpips
import math
import time 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_results_summary(results):
    """Print a summary of evaluation results"""
    print("\nResults Summary:")
    
    # Image task results
    if "category_scores" in results["image_tasks"]:
        print("\n=== Image Task Results ===")
        print(f"Total samples: {results['image_tasks']['total_samples']}")
        print(f"Processed samples: {results['image_tasks']['processed_samples']}")
        print(f"Skipped samples: {results['image_tasks']['skipped_samples']}")
        
        for category, metrics in results["image_tasks"]["category_scores"].items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Video task results
    if "image_to_video" in results:
        i2v = results["image_to_video"]
        print("\n=== Image-to-Video Generation ===")
        print(f"Total samples: {i2v['total_samples']}")
        print(f"Processed samples: {i2v['processed_samples']}")
        print(f"Skipped samples: {i2v['skipped_samples']}")
        if "average_score" in i2v:
            print(f"Average Score: {i2v['average_score']:.4f}")
    
    if "text_to_video" in results:
        t2v = results["text_to_video"]
        print("\n=== Text-to-Video Generation ===")
        print(f"Total samples: {t2v['total_samples']}")
        print(f"Processed samples: {t2v['processed_samples']}")
        print(f"Skipped samples: {t2v['skipped_samples']}")
        if "average_score" in t2v:
            print(f"Average Score: {t2v['average_score']:.4f}")
            
    if "video_prediction" in results:
        vp = results["video_prediction"]
        print("\n=== Video Prediction ===")
        print(f"Total samples: {vp['total_samples']}")
        print(f"Processed samples: {vp['processed_samples']}")
        print(f"Skipped samples: {vp['skipped_samples']}")
        if "average_score" in vp:
            print(f"Average Score: {vp['average_score']:.4f}")

def evaluate_image_tasks(data: List[Dict], base_path: str, clip_model_path: str):
    """Evaluate image-related tasks (reconstruction, editing, generation)"""
    evaluator = ImageEvaluator(clip_model_path, base_path)
    
    # Group results by category
    results = {
        "total_samples": len(data),
        "processed_samples": 0,
        "skipped_samples": 0,
        "categories": defaultdict(list)
    }
    
    # Process each sample
    for item in tqdm(data, desc="Processing image tasks"):
        category = item.get('category', '')
        
        # Skip items with error flag
        error_flag = item.get('error', 0)
        if error_flag != 0:
            results["skipped_samples"] += 1
            continue

        # Get output image path
        if 'output' not in item or not item['output']:
            results["skipped_samples"] += 1
            continue
            
        if isinstance(item['output'], dict) and 'output_image' in item['output']:
            output_path = item['output']['output_image']
        elif isinstance(item['output'], str):
            output_path = item['output']
        else:
            results["skipped_samples"] += 1
            continue
                    
        try:
            # Process based on category
            if category == 'Fine-Grained_Image_Reconstruction':
                target_path = item['data']['image']
                result = evaluator.evaluate_reconstruction(output_path, target_path)
                if result:
                    results["categories"][category].append(result)
                    results["processed_samples"] += 1
                    
            elif category == 'Text-Image_Editing':
                target_path = item['data']['edited_image']
                prompt = item.get('Text_Prompt', '')
                result = evaluator.evaluate_editing(output_path, target_path, prompt)
                if result:
                    results["categories"][category].append(result)
                    results["processed_samples"] += 1
                    
            elif category == 'Text-Image_Generation':
                target_path = item['data']['image']
                prompt = item.get('Text_Prompt', '')
                result = evaluator.evaluate_generation(output_path, target_path, prompt)
                if result:
                    results["categories"][category].append(result)
                    results["processed_samples"] += 1
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            logger.error(f"Category: {category}")
            logger.error(f"Output path: {output_path}")
            results["skipped_samples"] += 1
    
    # Calculate average scores per category
    category_scores = {}
    for category, items in results["categories"].items():
        if not items:
            continue
            
        metrics = {}
        for metric in items[0].keys():
            avg_value = np.mean([r[metric] for r in items])
            metrics[metric] = avg_value
        
        category_scores[category] = metrics
    
    results["category_scores"] = category_scores
    
    return results
        

    def process_dataset(self, json_path: str, mp4_prefix: str) -> Dict:
        logger.info(f"Loading data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        i2v_results = self.process_image_to_video(data, mp4_prefix)
        t2v_results = self.process_text_to_video(data, mp4_prefix)
        vp_results = self.process_video_prediction(data, mp4_prefix)
        
        results = {
            "image_to_video": i2v_results,
            "text_to_video": t2v_results,
            "video_prediction": vp_results
        }
        
        return results

class ImageEvaluator:
    def __init__(self, clip_model_path, base_path="/data/xwl/xwl_data/decode_images", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        
        self.base_path = base_path
        
    def load_and_preprocess_image(self, image_path: str, is_output: bool = False) -> torch.Tensor:
        """Load and preprocess image"""
        if not image_path:
            return None
            
        # Determine path based on whether it's an output image
        full_path = image_path if is_output else os.path.join(self.base_path, image_path.lstrip('/'))
        
        if not os.path.exists(full_path):
            logger.warning(f"Warning: Image path does not exist: {full_path}")
            return None
            
        try:
            img = Image.open(full_path).convert('RGB')
            return transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {str(e)}")
            return None

    def resize_images(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize images to match each other"""
        if img1.shape != img2.shape:
            # Get target size (use smaller dimensions)
            target_size = (
                min(img1.shape[2], img2.shape[2]),
                min(img1.shape[3], img2.shape[3])
            )
            
            # Resize both images to the same size
            img1 = interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
            img2 = interpolate(img2, size=target_size, mode='bilinear', align_corners=False)
            
        return img1, img2

    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate LPIPS value and return 1-LPIPS as per requirement"""
        # Make sure images are the same size
        img1, img2 = self.resize_images(img1, img2)
        with torch.no_grad():
            lpips_value = self.lpips_model(img1, img2).item() * 100
            # Return 1-LPIPS as per requirement
            return 100.0 - lpips_value

    def calculate_clip_similarity(self, img: torch.Tensor, text: str = None) -> float:
        """Calculate CLIP similarity"""
        with torch.no_grad():
            # Convert tensor to PIL Image
            img_pil = transforms.ToPILImage()(img.squeeze(0))
            
            if text:
                # Process image and text, setting max length to 77
                inputs = self.clip_processor(
                    images=img_pil,
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    max_length=77,  # CLIP's max text length
                    truncation=True  # Truncate if too long
                ).to(self.device)
                
                # Get image and text features
                outputs = self.clip_model(**inputs)
                
                # Calculate similarity
                logits_per_image = outputs.logits_per_image
                similarity = logits_per_image[0][0].item()
                return similarity
            else:
                # Process just the image
                inputs = self.clip_processor(
                    images=img_pil,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get image features
                image_features = self.clip_model.get_image_features(**inputs)
                return image_features

    def evaluate_reconstruction(self, output_path: str, target_path: str) -> Dict:
        """Evaluate image reconstruction task using only 1-LPIPS"""
        output_img = self.load_and_preprocess_image(output_path, is_output=True)
        target_img = self.load_and_preprocess_image(target_path, is_output=False)
        
        if output_img is None or target_img is None:
            return None
        
        # Calculate 1-LPIPS (higher is better)
        lpips_score = self.calculate_lpips(output_img, target_img)
        
        # Using only 1-LPIPS as the score
        avg_score = lpips_score
            
        return {
            'lpips': lpips_score,
            'score': avg_score
        }

    def evaluate_editing(self, output_path: str, target_path: str, prompt: str) -> Dict:
        """Evaluate image editing task using CLIP-I and CLIP-T average"""
        output_img = self.load_and_preprocess_image(output_path, is_output=True)
        target_img = self.load_and_preprocess_image(target_path, is_output=False)
        
        if output_img is None or target_img is None:
            return None
            
        # Resize images to match
        output_img, target_img = self.resize_images(output_img, target_img)
            
        # Calculate image-to-image CLIP similarity
        output_features = self.calculate_clip_similarity(output_img)
        target_features = self.calculate_clip_similarity(target_img)
        clip_i = torch.cosine_similarity(output_features, target_features, dim=1).item() * 100
        
        # Calculate image-to-text CLIP similarity
        clip_t = self.calculate_clip_similarity(output_img, prompt) 
        
        # Average of CLIP-I and CLIP-T as the score
        avg_score = (clip_i + clip_t) / 2.0
        
        return {
            'clip_i': clip_i,
            'clip_t': clip_t,
            'score': avg_score
        }

    def evaluate_generation(self, output_path: str, target_path: str, prompt: str) -> Dict:
        """Evaluate image generation task"""
        output_img = self.load_and_preprocess_image(output_path, is_output=True)
        target_img = self.load_and_preprocess_image(target_path, is_output=False)
        
        if output_img is None or target_img is None:
            return None
            
        # Resize images to match
        output_img, target_img = self.resize_images(output_img, target_img)
            
        # Calculate image-to-image CLIP similarity
        output_features = self.calculate_clip_similarity(output_img)
        target_features = self.calculate_clip_similarity(target_img)
        clip_i = torch.cosine_similarity(output_features, target_features, dim=1).item() * 100
        
        # Calculate image-to-text CLIP similarity
        clip_t = self.calculate_clip_similarity(output_img, prompt)
        
        # Average of CLIP-I and CLIP-T as the score
        avg_score = (clip_i   + clip_t) / 2.0
        
        return {
            'clip_i': clip_i,
            'clip_t': clip_t,
            'score': avg_score
        }


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for FID calculation"""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.backbone = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2, inception.Mixed_5b,
            inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d,
            inception.Mixed_6e, inception.Mixed_7a,
            inception.Mixed_7b, inception.Mixed_7c
        )
        self.backbone.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))
        return features.reshape(features.shape[0], -1)


class ResNetFeatureExtractor(nn.Module):
    """ResNet feature extractor for FVD calculation"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, 2048)
        features = features.mean(dim=1)
        return features


class CLIPMetrics:
    """Class for calculating CLIP similarity"""
    def __init__(self, clip_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        
    def calculate_similarity(self, image_path: str, text_prompt: str) -> float:
        try:
            image = Image.open(image_path).convert('RGB')
            # Handle text length limitation
            if len(text_prompt.split()) > 70:
                logger.warning(f"Text prompt too long, truncating: {text_prompt[:100]}...")
                text_prompt = ' '.join(text_prompt.split()[:70])
            
            inputs = self.processor(
                images=image,
                text=[text_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                return float(similarity[0])
        except Exception as e:
            logger.error(f"Error calculating CLIP similarity for {image_path}: {e}")
            return None


def extract_keyframes(video_path: str, 
                      output_dir: str = None, 
                      num_frames: int = 16, 
                      min_scene_change: float = 20.0,
                      use_uniform_fallback: bool = True) -> List[str]:
    """
    Extract keyframes from MP4 video. Attempts to detect scene changes, and falls back to uniform sampling if needed.
    
    Args:
        video_path (str): Path to MP4 file
        output_dir (str, optional): Directory to save keyframes. If None, creates folder in video directory
        num_frames (int, optional): Number of frames to extract, default is 16
        min_scene_change (float, optional): Minimum scene change threshold, higher means fewer detected changes. Default is 20.0
        use_uniform_fallback (bool, optional): Whether to use uniform sampling if not enough keyframes are detected. Default is True
        
    Returns:
        List[str]: List of paths to extracted keyframe images
    """
    # Check if video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    # Prepare output directory
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, f"{video_name}_keyframes")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")
    
    logger.info(f"Video {video_path} with FPS: {fps}, total frames: {frame_count}")
    
    # Initialize variables
    prev_frame = None
    keyframes = []
    keyframe_indices = []
    frame_idx = 0
    
    # Use frame difference method to detect scene changes
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame is always a keyframe
        if prev_frame is None:
            keyframe_path = os.path.join(output_dir, f"keyframe_{frame_idx:04d}.jpg")
            cv2.imwrite(keyframe_path, frame)
            keyframes.append(keyframe_path)
            keyframe_indices.append(frame_idx)
            prev_frame = gray
        else:
            # Calculate difference between current and previous frame
            frame_diff = cv2.absdiff(gray, prev_frame)
            # Calculate average difference value (0-255)
            mean_diff = np.mean(frame_diff)
            
            # If difference exceeds threshold, consider it a scene change
            if mean_diff > min_scene_change:
                keyframe_path = os.path.join(output_dir, f"keyframe_{frame_idx:04d}.jpg")
                cv2.imwrite(keyframe_path, frame)
                keyframes.append(keyframe_path)
                keyframe_indices.append(frame_idx)
                
            prev_frame = gray
        
        frame_idx += 1
    
    # Release video resources
    cap.release()
    
    # If not enough keyframes detected and uniform sampling is allowed
    if len(keyframes) < num_frames and use_uniform_fallback:
        logger.info(f"Detected {len(keyframes)} keyframes using scene change detection, less than {num_frames}, will use uniform sampling")
        
        # Clear already detected keyframes
        keyframes = []
        keyframe_indices = []
        
        # Reopen video
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames uniformly
        sample_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in sample_indices:
                keyframe_path = os.path.join(output_dir, f"keyframe_{current_frame:04d}.jpg")
                cv2.imwrite(keyframe_path, frame)
                keyframes.append(keyframe_path)
                keyframe_indices.append(current_frame)
                
            current_frame += 1
                
        cap.release()
    # If too many keyframes detected, select uniformly
    elif len(keyframes) > num_frames:
        logger.info(f"Detected {len(keyframes)} keyframes using scene change detection, more than {num_frames}, will select uniformly")
        
        selected_indices = np.linspace(0, len(keyframes) - 1, num_frames, dtype=int)
        keyframes = [keyframes[i] for i in selected_indices]
        keyframe_indices = [keyframe_indices[i] for i in selected_indices]
    
    logger.info(f"Finally extracted {len(keyframes)} keyframes")
    
    # Print keyframe positions in video (timestamps)
    if fps > 0:
        keyframe_times = [idx / fps for idx in keyframe_indices]
        for i, (idx, time) in enumerate(zip(keyframe_indices, keyframe_times)):
            logger.info(f"Keyframe {i+1}: Frame {idx}, Time {time:.2f}s")
    
    return keyframes


class VideoMetricsCalculator:
    def __init__(self, clip_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fvd_extractor = ResNetFeatureExtractor().to(self.device)
        self.inception = InceptionV3FeatureExtractor().to(self.device)
        self.clip_metrics = CLIPMetrics(clip_model_path)
        
        # Transform for FVD
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for FID
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 临时目录，用于存储从MP4提取出的关键帧
        self.temp_dir = "temp_frames"
        os.makedirs(self.temp_dir, exist_ok=True)

    def _validate_item(self, item: Dict) -> bool:
        """验证图像到视频生成项目是否有效"""
        if 'data' not in item:
            return False
        if 'image' not in item['data'] or 'video' not in item['data']:
            return False
        if 'output' not in item or not item['output']:
            return False
        if 'Text_Prompt' not in item:
            return False
        return True

    def _validate_t2v_item(self, item: Dict) -> bool:
        """验证文本到视频生成项目是否有效"""
        if 'data' not in item or 'image' not in item['data']:
            logger.warning("Missing data or video field")
            return False
        if 'output' not in item or not item['output']:
            logger.warning("Missing or empty output field")
            return False
        if 'Text_Prompt' not in item:
            logger.warning("Missing Text_Prompt field")
            return False
        return True

    def _validate_features(self, real_features: np.ndarray, gen_features: np.ndarray) -> bool:
        """验证特征向量是否有效"""
        if real_features.ndim != 2 or gen_features.ndim != 2:
            return False
        if real_features.shape[1] != 2048 or gen_features.shape[1] != 2048:
            return False
        if not (np.isfinite(real_features).all() and np.isfinite(gen_features).all()):
            return False
        return True

    def sample_frames_from_mp4(self, mp4_path: str, num_frames: int = 16, temp_path: str = None) -> List[str]:
        """
        从MP4文件中均匀采样指定数量的帧，并保存为图像文件
        
        Args:
            mp4_path: MP4文件路径
            num_frames: 要采样的帧数量
            temp_path: 临时存储帧的目录，如果为None则使用当前时间创建临时目录
            
        Returns:
            采样帧的文件路径列表
        """
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(f"MP4 file not found: {mp4_path}")
            
        # 创建临时目录
        if temp_path is None:
            temp_path = os.path.join(self.temp_dir, f"{int(time.time())}_{os.path.basename(mp4_path).split('.')[0]}")
        os.makedirs(temp_path, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {mp4_path}")
            
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames: {mp4_path}")
            
        # 均匀采样帧
        sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frame_paths = []
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in sample_indices:
                frame_path = os.path.join(temp_path, f"frame_{current_frame:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            current_frame += 1
            
        cap.release()
        
        if len(frame_paths) != num_frames:
            logger.warning(f"Could only extract {len(frame_paths)} frames from {mp4_path}, requested {num_frames}")
            
        return frame_paths

    def extract_frames_for_clip(self, mp4_path: str, num_frames: int = 8) -> List[str]:
        """提取视频帧用于CLIP分析"""
        return self.sample_frames_from_mp4(mp4_path, num_frames)

    def load_frame_sequence(self, frame_paths: List[str]) -> torch.Tensor:
        """加载图像帧序列为张量"""
        frames = []
        for path in frame_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
                frames.append(img_tensor)
            except Exception as e:
                logger.error(f"Error loading frame {path}: {e}")
                continue
        
        if not frames:
            raise ValueError("No frames were successfully loaded")
            
        video_tensor = torch.stack(frames)
        return video_tensor.permute(1, 0, 2, 3)
    
    def load_and_process_mp4(self, video_path: str, target_frames: int = 16) -> Tuple[torch.Tensor, List[str]]:
        """加载MP4文件，提取并处理帧"""
        # 采样帧并获取帧路径
        frame_paths = self.sample_frames_from_mp4(video_path, target_frames)
        
        # 加载帧序列
        video_tensor = self.load_frame_sequence(frame_paths)
        
        return video_tensor, frame_paths

    def extract_features(self, video_tensor: torch.Tensor) -> np.ndarray:
        """提取视频特征"""
        video_tensor = video_tensor.to(self.device)
        with torch.no_grad():
            if video_tensor.dim() == 4:
                video_tensor = video_tensor.unsqueeze(0)
            features = self.fvd_extractor(video_tensor)
        return features.cpu().numpy()

    def extract_inception_features(self, images: List[str], batch_size: int = 32) -> np.ndarray:
        """提取图像的Inception特征"""
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch_paths = images[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.inception_transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            batch = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_features = self.inception(batch)
                features_list.append(batch_features.cpu().numpy())
        
        if not features_list:
            raise ValueError("No features were extracted")
            
        return np.concatenate(features_list, axis=0)

    def calculate_fid(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """
        计算FID分数
        """
        try:
            if real_features.ndim == 1:
                real_features = real_features.reshape(1, -1)
            if gen_features.ndim == 1:
                gen_features = gen_features.reshape(1, -1)
            
            if real_features.shape[0] == 1 or gen_features.shape[0] == 1:
                mu1 = np.mean(real_features, axis=0)
                mu2 = np.mean(gen_features, axis=0)
                return float(np.sum((mu1 - mu2) ** 2))
            
            mu1 = np.mean(real_features, axis=0)
            mu2 = np.mean(gen_features, axis=0)
            
            sigma1 = np.cov(real_features, rowvar=False)
            sigma2 = np.cov(gen_features, rowvar=False)
            
            diff = mu1 - mu2
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid)
            
        except Exception as e:
            logger.error(f"Error in FID calculation: {e}")
            return float('inf')

    def calculate_fvd(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """计算FVD分数（与FID计算方法相同）"""
        return self.calculate_fid(real_features, gen_features)

    def normalize_fvd(self, fvd_score: float, max_value: float = 1000.0) -> float:
        """将FVD分数归一化到0-1范围"""
        norm_value = 1.0 - (fvd_score / max_value)
        return max(0.0, min(1.0, norm_value))

    def process_image_to_video(self, data: List[Dict], mp4_prefix: str) -> Dict:
        """处理图像到视频生成任务"""
        i2v_data = [item for item in data if item.get('category') == 'Conditional_Image_to_Video_Generation']
        
        results = {
            "total_samples": len(i2v_data),
            "processed_samples": 0,
            "skipped_samples": 0,
            "sample_scores": []
        }
        
        for idx, item in tqdm(enumerate(i2v_data), total=len(i2v_data)):
            try:
                if not self._validate_item(item):
                    # 添加零分记录
                    results["sample_scores"].append({
                        "id": item.get('id', 'N/A'),
                        "fvd": float('inf'),
                        "norm_fvd": 0.0,
                        "clip_t": 0.0,
                        "score": 0.0
                    })
                    results["skipped_samples"] += 1
                    continue
                
                # 获取生成视频路径和参考视频路径
                gen_video_path = item['output']  # 假设output字段现在是MP4文件路径
                if not os.path.isabs(gen_video_path):
                    gen_video_path = os.path.join(mp4_prefix, gen_video_path.lstrip('/'))
                    
                ref_video_path = os.path.join(mp4_prefix, item['data']['video'].lstrip('/'))
                text_prompt = item.get('Text_Prompt', '')
                
                # 处理每个样本
                try:
                    # 从生成视频中提取帧用于CLIP评分
                    gen_frames = self.extract_frames_for_clip(gen_video_path)
                    
                    # 计算每一帧的CLIP-T分数
                    frame_clip_scores = []
                    for frame_path in gen_frames:
                        clip_score = self.clip_metrics.calculate_similarity(frame_path, text_prompt)
                        if clip_score is not None:
                            frame_clip_scores.append(clip_score)
                    
                    # 加载并处理视频张量
                    gen_video_tensor, _ = self.load_and_process_mp4(gen_video_path)
                    ref_video_tensor, _ = self.load_and_process_mp4(ref_video_path)
                    
                    # 提取特征
                    gen_features = self.extract_features(gen_video_tensor)
                    ref_features = self.extract_features(ref_video_tensor)
                    
                    # 计算FVD分数
                    if self._validate_features(ref_features, gen_features) and frame_clip_scores:
                        fvd_score = float(self.calculate_fvd(ref_features, gen_features))
                        
                        # 归一化FVD
                        norm_fvd = self.normalize_fvd(fvd_score)
                        
                        # 计算平均CLIP-T分数
                        avg_clip_t = np.mean(frame_clip_scores)
                        
                        # 最终得分是归一化FVD和CLIP-T的平均值
                        final_score = (norm_fvd + avg_clip_t) / 2.0
                        
                        # 存储样本得分
                        results["sample_scores"].append({
                            "id": item.get('id', 'N/A'),
                            "fvd": fvd_score,
                            "norm_fvd": norm_fvd,
                            "clip_t": avg_clip_t,
                            "score": final_score
                        })
                        
                        results["processed_samples"] += 1
                    else:
                        # 无效特征或无CLIP分数
                        results["sample_scores"].append({
                            "id": item.get('id', 'N/A'),
                            "fvd": float('inf'),
                            "norm_fvd": 0.0,
                            "clip_t": 0.0,
                            "score": 0.0
                        })
                        results["skipped_samples"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing video: {e}")
                    # 处理失败时添加零分
                    results["sample_scores"].append({
                        "id": item.get('id', 'N/A'),
                        "fvd": float('inf'),
                        "norm_fvd": 0.0,
                        "clip_t": 0.0,
                        "score": 0.0
                    })
                    results["skipped_samples"] += 1
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                # 添加零分
                results["sample_scores"].append({
                    "id": item.get('id', 'N/A'),
                    "fvd": float('inf'),
                    "norm_fvd": 0.0,
                    "clip_t": 0.0,
                    "score": 0.0
                })
                results["skipped_samples"] += 1
                continue
        
        # 计算总体平均得分
        results["average_score"] = np.mean([s["score"] for s in results["sample_scores"]]) if results["sample_scores"] else 0.0
        
        return results

    def process_text_to_video(self, data: List[Dict], mp4_prefix: str) -> Dict:
        """处理文本到视频生成任务"""
        t2v_data = [item for item in data if item.get('category') == 'Text-to-Video_Generation']
        
        results = {
            "total_samples": len(t2v_data),
            "processed_samples": 0,
            "skipped_samples": 0,
            "sample_scores": []
        }
        
        for idx, item in tqdm(enumerate(t2v_data), total=len(t2v_data)):
            try:
                if not self._validate_t2v_item(item):
                    # 添加零分记录
                    results["sample_scores"].append({
                        "id": item.get('id', 'N/A'),
                        "clip_t": 0.0,
                        "score": 0.0
                    })
                    results["skipped_samples"] += 1
                    continue
                
                # 获取生成视频路径
                gen_video_path = item['output']  # 假设output字段现在是MP4文件路径
                if not os.path.isabs(gen_video_path):
                    gen_video_path = os.path.join(mp4_prefix, gen_video_path.lstrip('/'))
                    
                text_prompt = item.get('Text_Prompt', '')
                
                try:
                    # 从生成视频中提取帧
                    gen_frames = self.extract_frames_for_clip(gen_video_path)
                    
                    # 计算每一帧的CLIP-T分数
                    frame_clip_scores = []
                    for frame_path in gen_frames:
                        clip_score = self.clip_metrics.calculate_similarity(frame_path, text_prompt)
                        if clip_score is not None:
                            frame_clip_scores.append(clip_score)
                    
                    # 计算平均CLIP-T分数
                    if frame_clip_scores:
                        avg_clip_t = np.mean(frame_clip_scores)
                        
                        # 对于文本到视频任务，得分就是CLIP-T分数
                        results["sample_scores"].append({
                            "id": item.get('id', 'N/A'),
                            "clip_t": avg_clip_t,
                            "score": avg_clip_t
                        })
                        
                        results["processed_samples"] += 1
                    else:
                        # 无CLIP分数时添加零分
                        results["sample_scores"].append({
                            "id": item.get('id', 'N/A'),
                            "clip_t": 0.0,
                            "score": 0.0
                        })
                        results["skipped_samples"] += 1
                        
                except Exception as e:
                    logger.error(f"Error extracting frames from video: {e}")
                    # 处理失败时添加零分
                    results["sample_scores"].append({
                        "id": item.get('id', 'N/A'),
                        "clip_t": 0.0,
                        "score": 0.0
                    })
                    results["skipped_samples"] += 1
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                # 添加零分
                results["sample_scores"].append({
                    "id": item.get('id', 'N/A'),
                    "clip_t": 0.0,
                    "score": 0.0
                })
                results["skipped_samples"] += 1
                continue
        
        # 计算总体平均得分
        results["average_score"] = np.mean([s["score"] for s in results["sample_scores"]]) if results["sample_scores"] else 0.0
        
        return results

    def process_video_prediction(self, data: List[Dict], mp4_prefix: str) -> Dict:
        """处理视频预测任务"""
        vp_data = [item for item in data if item.get('category') == 'Video prediction']
        
        results = {
            "total_samples": len(vp_data),
            "processed_samples": 0,
            "skipped_samples": 0,
            "sample_scores": [],
            "fid_scores": []  # 存储FID分数
        }
        
        for idx, item in tqdm(enumerate(vp_data), total=len(vp_data)):
            try:
                if not self._validate_item(item):
                    # 添加零分记录
                    results["sample_scores"].append({
                        "id": item.get('id', 'N/A'),
                        "fvd": float('inf'),
                        "norm_fvd": 0.0,
                        "fid": float('inf'),
                        "norm_fid": 0.0,
                        "score": 0.0
                    })
                    results["skipped_samples"] += 1
                    continue
                
                # 获取生成视频路径和参考视频路径
                gen_video_path = item['output']  # 假设output字段现在是MP4文件路径
                if not os.path.isabs(gen_video_path):
                    gen_video_path = os.path.join(mp4_prefix, gen_video_path.lstrip('/'))
                    
                ref_video_path = os.path.join(mp4_prefix, item['data']['video'].lstrip('/'))
                source_image_path = os.path.join(mp4_prefix, item['data']['image'].lstrip('/'))
                
                try:
                    # 提取源图像特征用于FID计算
                    source_image_features = self.extract_inception_features_batch([source_image_path])
                    
                    # 从生成视频中提取帧
                    gen_frames = self.extract_frames_for_clip(gen_video_path, num_frames=16)
                    
                    # 从参考视频中提取相同数量的帧，用于FID比较
                    ref_frames = self.sample_frames_from_mp4(ref_video_path, num_frames=len(gen_frames))
                    
                    # 计算FID - 比较参考视频帧与生成帧
                    ref_frames_features = self.extract_inception_features_batch(ref_frames)
                    gen_frames_features = self.extract_inception_features_batch(gen_frames)
                    
                    fid_score = self.calculate_fid(ref_frames_features, gen_frames_features)
                    norm_fid = self.normalize_fid(fid_score)
                    results["fid_scores"].append(fid_score)
                    
                    # 加载并处理视频张量用于FVD计算
                    gen_video_tensor, _ = self.load_and_process_mp4(gen_video_path)
                    ref_video_tensor, _ = self.load_and_process_mp4(ref_video_path)
                    
                    # 提取特征
                    gen_features = self.extract_features(gen_video_tensor)
                    ref_features = self.extract_features(ref_video_tensor)
                    
                    # 计算FVD分数
                    if self._validate_features(ref_features, gen_features):
                        fvd_score = float(self.calculate_fvd(ref_features, gen_features))
                        
                        # 归一化FVD
                        norm_fvd = self.normalize_fvd(fvd_score)
                        
                        # 最终得分是归一化FVD和归一化FID的平均值
                        final_score = (norm_fvd + norm_fid) / 2.0
                        
                        # 存储样本得分
                        results["sample_scores"].append({
                            "id": item.get('id', 'N/A'),
                            "fvd": fvd_score,
                            "norm_fvd": norm_fvd,
                            "fid": fid_score,
                            "norm_fid": norm_fid,
                            "score": final_score
                        })
                        
                        results["processed_samples"] += 1
                    else:
                        # 如果FVD无效但FID有效
                        if fid_score != float('inf'):
                            results["sample_scores"].append({
                                "id": item.get('id', 'N/A'),
                                "fvd": float('inf'),
                                "norm_fvd": 0.0,
                                "fid": fid_score,
                                "norm_fid": norm_fid,
                                "score": norm_fid  # 仅使用FID评分
                            })
                            results["processed_samples"] += 1
                        else:
                            # 无效特征
                            results["sample_scores"].append({
                                "id": item.get('id', 'N/A'),
                                "fvd": float('inf'),
                                "norm_fvd": 0.0,
                                "fid": float('inf'),
                                "norm_fid": 0.0,
                                "score": 0.0
                            })
                            results["skipped_samples"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing video pair: {e}")
                    # 处理失败时添加零分
                    results["sample_scores"].append({
                        "id": item.get('id', 'N/A'),
                        "fvd": float('inf'),
                        "norm_fvd": 0.0,
                        "fid": float('inf'),
                        "norm_fid": 0.0,
                        "score": 0.0
                    })
                    results["skipped_samples"] += 1
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                # 添加零分
                results["sample_scores"].append({
                    "id": item.get('id', 'N/A'),
                    "fvd": float('inf'),
                    "norm_fvd": 0.0,
                    "fid": float('inf'),
                    "norm_fid": 0.0,
                    "score": 0.0
                })
                results["skipped_samples"] += 1
                continue
        
        # 计算平均FID
        if results["fid_scores"]:
            results["avg_fid"] = float(np.mean(results["fid_scores"]))
            results["avg_norm_fid"] = float(np.mean([self.normalize_fid(s) for s in results["fid_scores"]]))
        
        # 计算总体平均得分
        results["average_score"] = np.mean([s["score"] for s in results["sample_scores"]]) if results["sample_scores"] else 0.0
        
        return results

    def process_dataset(self, json_path: str, mp4_prefix: str) -> Dict:
        """处理整个数据集"""
        logger.info(f"Loading data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 处理各个任务
        i2v_results = self.process_image_to_video(data, mp4_prefix)
        t2v_results = self.process_text_to_video(data, mp4_prefix)
        vp_results = self.process_video_prediction(data, mp4_prefix)
        
        # 组合结果
        results = {
            "image_to_video": i2v_results,
            "text_to_video": t2v_results,
            "video_prediction": vp_results
        }
        
        return results
        
    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")

def calculate_task_scores(results_dict):
    """
    计算所有任务的得分，每个任务的得分是其所有指标的平均值，总得分是所有任务得分的平均值
    
    Args:
        results_dict: 包含所有评估结果的字典
    
    Returns:
        带有每个任务得分和总体得分的字典
    """
    # 定义所有预期任务
    image_tasks = [
        'Fine-Grained_Image_Reconstruction',
        'Text-Image_Editing',
        'Text-Image_Generation'
    ]
    
    video_tasks = [
        'image_to_video',    # 对应 Conditional_Image_to_Video_Generation
        'text_to_video',     # 对应 Text-to-Video_Generation
        'video_prediction'   # 对应 Video prediction
    ]
    
    task_scores = {}
    all_scores = []
    
    # 处理图像任务
    if "image_tasks" in results_dict and "category_scores" in results_dict["image_tasks"]:
        image_scores = results_dict["image_tasks"]["category_scores"]
        
        # 检查每个预期的图像任务
        for task in image_tasks:
            if task in image_scores:
                # 收集该任务的所有指标（除了'score'，因为我们要自己计算）
                metrics = []
                for key, value in image_scores[task].items():
                    if key != 'score':  # 排除已有的'score'，我们要重新计算
                        metrics.append(value)
                
                # 如果有指标，计算平均得分；否则为0
                if metrics:
                    score = sum(metrics) / len(metrics)
                else:
                    score = 0.0
            else:
                # 如果任务不存在，计为0分
                score = 0.0
            
            # 记录任务得分
            task_scores[task] = score
            all_scores.append(score)
    else:
        # 如果图像任务结果不存在，所有图像任务得分为0
        for task in image_tasks:
            task_scores[task] = 0.0
            all_scores.append(0.0)
    
    # 处理视频任务
    for task in video_tasks:
        if task in results_dict and "average_score" in results_dict[task]:
            score = results_dict[task]["average_score"]
        else:
            # 如果任务不存在或没有得分，计为0分
            score = 0.0
        
        # 记录任务得分
        task_scores[task] = score
        all_scores.append(score)
    
    # 计算所有任务的平均得分作为总分
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return {
        "task_scores": task_scores,
        "generation_score": overall_score
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multimodal generation tasks')
    parser.add_argument('--json_path', required=True, help='Path to the JSON file with results')
    parser.add_argument('--base_path', default="/data/xwl/xwl_data/decode_images", 
                        help='Base path for image and video files')
    parser.add_argument('--clip_model_path', default="/data/xwl/xwl_data/.cache/clip-vit-large-patch14", 
                        help='Path to CLIP model')
    parser.add_argument('--output_file', default="evaluation_results.json", 
                        help='Where to save the results')
    args = parser.parse_args()
    
    all_results = {}
    
    try:
        # 读取输入数据
        logger.info(f"Loading data from {args.json_path}")
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        
        # 评估图像任务
        try:
            logger.info("Evaluating image tasks...")
            image_results = evaluate_image_tasks(data, args.base_path, args.clip_model_path)
            all_results["image_tasks"] = image_results
        except Exception as e:
            logger.error(f"Error in image task evaluation: {e}", exc_info=True)
            # 创建空的图像结果，所有预期的图像任务得分为0
            all_results["image_tasks"] = {
                "total_samples": 0,
                "processed_samples": 0,
                "skipped_samples": 0,
                "categories": {},
                "category_scores": {
                    "Fine-Grained_Image_Reconstruction": {"lpips": 0.0},
                    "Text-Image_Editing": {"clip_i": 0.0, "clip_t": 0.0},
                    "Text-Image_Generation": {"clip_i": 0.0, "clip_t": 0.0}
                }
            }
        
        # 评估视频任务
        try:
            logger.info("Evaluating video tasks...")
            video_calculator = VideoMetricsCalculator(args.clip_model_path)
            video_results = video_calculator.process_dataset(args.json_path, args.base_path)
            
            # 如果video_results为None，初始化为空字典
            if video_results is None:
                video_results = {}
                
            # 确保所有视频任务都有结果，没有的设为0分
            for task in ['image_to_video', 'text_to_video', 'video_prediction']:
                if task not in video_results:
                    video_results[task] = {
                        "total_samples": 0,
                        "processed_samples": 0,
                        "skipped_samples": 0,
                        "sample_scores": [],
                        "average_score": 0.0
                    }
            
            # 更新结果
            all_results.update(video_results)
        except Exception as e:
            logger.error(f"Error in video task evaluation: {e}", exc_info=True)
            # 所有视频任务得分为0
            all_results.update({
                "image_to_video": {
                    "total_samples": 0,
                    "processed_samples": 0,
                    "skipped_samples": 0,
                    "sample_scores": [],
                    "average_score": 0.0
                },
                "text_to_video": {
                    "total_samples": 0,
                    "processed_samples": 0,
                    "skipped_samples": 0,
                    "sample_scores": [],
                    "average_score": 0.0
                },
                "video_prediction": {
                    "total_samples": 0,
                    "processed_samples": 0,
                    "skipped_samples": 0,
                    "sample_scores": [],
                    "average_score": 0.0
                }
            })
        
        # 计算所有任务的得分
        scores_result = calculate_task_scores(all_results)
        all_results["task_scores"] = scores_result["task_scores"]
        all_results["generation_score"] = scores_result["generation_score"]
        
        # 保存结果
        try:
            with open(args.output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        # 打印摘要
        try:
            print_results_summary(all_results)
            
            # 额外打印任务得分和总体得分
            print("\n=== Task Scores ===")
            for task, score in all_results["task_scores"].items():
                print(f"{task}: {score:.4f}")
            print(f"\nGeneration Score: {all_results['generation_score']:.4f}")
        except Exception as e:
            logger.error(f"Error printing results summary: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in evaluation: {e}", exc_info=True)
        
        # 创建基本的结果结构，所有任务得分为0
        empty_results = {
            "image_tasks": {
                "total_samples": 0,
                "processed_samples": 0,
                "skipped_samples": 0,
                "category_scores": {
                    "Fine-Grained_Image_Reconstruction": {"lpips": 0.0},
                    "Text-Image_Editing": {"clip_i": 0.0, "clip_t": 0.0},
                    "Text-Image_Generation": {"clip_i": 0.0, "clip_t": 0.0}
                }
            },
            "image_to_video": {
                "total_samples": 0,
                "processed_samples": 0,
                "skipped_samples": 0,
                "sample_scores": [],
                "average_score": 0.0
            },
            "text_to_video": {
                "total_samples": 0,
                "processed_samples": 0,
                "skipped_samples": 0,
                "sample_scores": [],
                "average_score": 0.0
            },
            "video_prediction": {
                "total_samples": 0,
                "processed_samples": 0,
                "skipped_samples": 0,
                "sample_scores": [],
                "average_score": 0.0
            }
        }
        
        # 计算任务得分
        scores_result = calculate_task_scores(empty_results)
        empty_results["task_scores"] = scores_result["task_scores"]
        empty_results["generation_score"] = scores_result["generation_score"]
        
        # 尝试保存
        try:
            with open(args.output_file, 'w') as f:
                json.dump(empty_results, f, indent=2)
            logger.info(f"Empty results saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to save empty results: {e}")

if __name__ == "__main__":
    main()