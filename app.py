#!/usr/bin/env python3
"""
PolyOCR Pipeline with EasyOCR-Level Accuracy
Complete production-ready OCR pipeline with JSON serialization fix
Version: 2.0
Author: AI Assistant
Date: 2025-08-02

Features:
- EasyOCR accuracy preservation
- Multi-language support (English, Hindi, Chinese, etc.)
- Structured output formats
- JSON serialization compatibility
- Bounding box visualization
- Comprehensive logging
- Title detection and formatting
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any
import warnings
import json
import re
from dataclasses import dataclass, asdict
import time
from datetime import datetime

import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure comprehensive logging
def setup_logging(log_file: str = "polyocr.log", log_level: str = "INFO"):
    """Setup logging configuration"""
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    logging.basicConfig(
        level=log_levels.get(log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class TextRegion:
    """Text region with comprehensive formatting information"""
    text: str
    bbox: List[List[int]]  # EasyOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    line_number: int = 0
    is_title: bool = False
    is_header: bool = False
    language: str = 'unknown'
    font_size: Optional[int] = None
    text_type: str = 'paragraph'  # paragraph, title, header, caption, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-safe types"""
        return {
            'text': str(self.text),
            'bbox': [[int(coord) for coord in point] for point in self.bbox],
            'confidence': float(self.confidence),
            'line_number': int(self.line_number),
            'is_title': bool(self.is_title),
            'is_header': bool(self.is_header),
            'language': str(self.language),
            'font_size': int(self.font_size) if self.font_size else None,
            'text_type': str(self.text_type)
        }

class ImagePreprocessor:
    """Optional image preprocessing utilities"""
    
    @staticmethod
    def enhance_image(image_path: str, output_path: Optional[str] = None) -> str:
        """Enhance image quality for better OCR (optional preprocessing)"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance contrast and sharpness
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                
                # Save enhanced image
                if output_path is None:
                    base_name = Path(image_path).stem
                    output_path = f"{base_name}_enhanced.jpg"
                
                img.save(output_path, 'JPEG', quality=95)
                logger.info(f"Enhanced image saved: {output_path}")
                return output_path
                
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image_path  # Return original if enhancement fails
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate image file"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                logger.error(f"Image file is empty: {image_path}")
                return False
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                img.verify()
            
            # Try to open with OpenCV
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                logger.error(f"Cannot read image with OpenCV: {image_path}")
                return False
            
            logger.info(f"Image validation passed: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

class EasyOCRWrapper:
    """
    EasyOCR wrapper that preserves exact standalone accuracy
    """
    
    def __init__(self, languages: List[str], gpu: Optional[bool] = None):
        self.languages = self._prepare_languages(languages)
        self.gpu_enabled = self._setup_gpu(gpu)
        self.reader = None
        self._initialize_reader()
    
    def _prepare_languages(self, languages: List[str]) -> List[str]:
        """Prepare and validate language codes for EasyOCR"""
        # Extended language mapping
        lang_map = {
            'en': 'en', 'english': 'en',
            'hi': 'hi', 'hindi': 'hi',
            'zh': 'ch_sim', 'chinese': 'ch_sim', 'zh-cn': 'ch_sim',
            'zh-tw': 'ch_tra', 'traditional_chinese': 'ch_tra',
            'ja': 'ja', 'japanese': 'ja',
            'ko': 'ko', 'korean': 'ko',
            'ar': 'ar', 'arabic': 'ar',
            'th': 'th', 'thai': 'th',
            'vi': 'vi', 'vietnamese': 'vi',
            'fr': 'fr', 'french': 'fr',
            'de': 'de', 'german': 'de',
            'es': 'es', 'spanish': 'es',
            'pt': 'pt', 'portuguese': 'pt',
            'ru': 'ru', 'russian': 'ru',
            'it': 'it', 'italian': 'it',
            'nl': 'nl', 'dutch': 'nl',
            'pl': 'pl', 'polish': 'pl',
            'tr': 'tr', 'turkish': 'tr',
            'sv': 'sv', 'swedish': 'sv',
            'da': 'da', 'danish': 'da',
            'no': 'no', 'norwegian': 'no',
            'fi': 'fi', 'finnish': 'fi'
        }
        
        easyocr_langs = []
        for lang in languages:
            lang_lower = lang.lower().strip()
            if lang_lower in lang_map:
                mapped = lang_map[lang_lower]
                if mapped not in easyocr_langs:
                    easyocr_langs.append(mapped)
            else:
                logger.warning(f"Unknown language code: {lang}, trying as-is")
                if lang_lower not in easyocr_langs:
                    easyocr_langs.append(lang_lower)
        
        if not easyocr_langs:
            logger.warning("No valid languages specified, defaulting to English")
            easyocr_langs = ['en']
        
        # EasyOCR stability requirement: Chinese variants need English
        chinese_variants = ['ch_sim', 'ch_tra']
        if any(lang in easyocr_langs for lang in chinese_variants) and 'en' not in easyocr_langs:
            easyocr_langs.append('en')
            logger.info("Added English for Chinese language stability")
        
        logger.info(f"Prepared languages for EasyOCR: {easyocr_langs}")
        return easyocr_langs
    
    def _setup_gpu(self, gpu: Optional[bool]) -> bool:
        """Setup GPU configuration"""
        if gpu is None:
            # Auto-detect GPU
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                try:
                    # Test GPU functionality
                    torch.cuda.empty_cache()
                    device_count = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(current_device)
                    logger.info(f"GPU detected: {device_name} (Device {current_device}/{device_count})")
                    return True
                except Exception as e:
                    logger.warning(f"GPU available but not functional: {e}")
                    return False
            else:
                logger.info("No GPU detected, using CPU")
                return False
        else:
            logger.info(f"GPU usage manually set to: {gpu}")
            return gpu and torch.cuda.is_available()
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader with optimal settings"""
        try:
            import easyocr
            
            logger.info(f"Initializing EasyOCR with GPU={self.gpu_enabled}")
            
            # Initialize with exact same settings as standalone EasyOCR
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu_enabled,
                verbose=False,  # Reduce console output
                download_enabled=True  # Allow model downloads if needed
            )
            
            logger.info(f"EasyOCR initialized successfully for languages: {self.languages}")
            
        except ImportError as e:
            error_msg = ("EasyOCR not installed. Install with:\n"
                        "pip install easyocr\n"
                        "or\n"
                        "conda install -c conda-forge easyocr")
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def extract_text_regions(self, image_path: str, 
                           confidence_threshold: float = 0.25,
                           paragraph_mode: bool = False,
                           width_ths: float = 0.7,
                           height_ths: float = 0.7) -> List[TextRegion]:
        """
        Extract text using EasyOCR with IDENTICAL processing to standalone usage
        """
        try:
            start_time = time.time()
            logger.info(f"Processing image: {image_path}")
            
            # Validate image first
            if not ImagePreprocessor.validate_image(image_path):
                return []
            
            # Use EasyOCR EXACTLY as you would standalone
            # No image preprocessing, no parameter modifications
            results = self.reader.readtext(
                image_path,
                detail=1,  # Get coordinates and confidence
                paragraph=paragraph_mode,
                width_ths=width_ths,
                height_ths=height_ths,
                decoder='greedy',  # Default decoder
                beamWidth=5,  # Default beam width
                batch_size=1,  # Default batch size
                workers=0,  # Use default worker count
                allowlist=None,  # No character filtering
                blocklist=None,  # No character blocking
                min_size=10,  # Minimum text size
                rotation_info=None  # No rotation correction
            )
            
            processing_time = time.time() - start_time
            logger.info(f"EasyOCR processing completed in {processing_time:.2f}s, found {len(results)} regions")
            
            # Convert to TextRegion objects with proper type conversion
            text_regions = []
            for i, result in enumerate(results):
                if len(result) >= 3:
                    bbox, text, confidence = result
                    
                    # Apply confidence threshold
                    if confidence >= confidence_threshold:
                        # Clean text minimally (preserve EasyOCR behavior)
                        cleaned_text = text.strip()
                        
                        if cleaned_text:  # Only include non-empty text
                            # Ensure JSON-safe bbox format
                            if isinstance(bbox, np.ndarray):
                                bbox = bbox.tolist()
                            elif isinstance(bbox, list):
                                bbox = [[int(coord) for coord in point] for point in bbox]
                            
                            # Detect language (basic detection)
                            detected_lang = self._detect_language(cleaned_text)
                            
                            region = TextRegion(
                                text=cleaned_text,
                                bbox=bbox,
                                confidence=float(confidence),
                                line_number=i,
                                language=detected_lang
                            )
                            text_regions.append(region)
                            logger.debug(f"Region {i}: '{cleaned_text[:50]}...' (conf: {confidence:.3f})")
            
            logger.info(f"Extracted {len(text_regions)} valid text regions above confidence threshold {confidence_threshold}")
            return text_regions
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _detect_language(self, text: str) -> str:
        """Basic language detection based on character patterns"""
        try:
            # Simple heuristic-based language detection
            if re.search(r'[\u4e00-\u9fff]', text):
                return 'chinese'
            elif re.search(r'[\u0900-\u097f]', text):
                return 'hindi'
            elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
                return 'japanese'
            elif re.search(r'[\uac00-\ud7af]', text):
                return 'korean'
            elif re.search(r'[\u0600-\u06ff]', text):
                return 'arabic'
            elif re.search(r'[\u0e00-\u0e7f]', text):
                return 'thai'
            elif re.search(r'[a-zA-Z]', text):
                return 'english'
            else:
                return 'unknown'
        except:
            return 'unknown'

class TextFormatter:
    """
    Advanced text formatting and structure detection
    """
    
    def __init__(self):
        self.title_patterns = [
            r'^[A-Z\s]{3,}$',  # All caps
            r'^\d+\.\s*[A-Z]',  # Numbered sections
            r'^[A-Z][^.!?]*:$',  # Ends with colon
            r'^(Chapter|Section|Part)\s+\d+',  # Chapter/Section headings
        ]
    
    def organize_regions(self, regions: List[TextRegion], 
                        image_width: int, image_height: int) -> List[TextRegion]:
        """
        Organize text regions into logical reading order with advanced formatting detection
        """
        if not regions:
            return regions
        
        logger.info(f"Organizing {len(regions)} text regions")
        
        # Sort regions by spatial position
        sorted_regions = self._spatial_sort(regions)
        
        # Group into lines and columns
        lines = self._group_into_lines(sorted_regions, image_height)
        
        # Detect text structure and formatting
        formatted_regions = []
        for line_num, line_regions in enumerate(lines):
            for region in line_regions:
                region.line_number = line_num
                
                # Advanced structure detection
                region.is_title = self._detect_title(region, line_num, len(lines))
                region.is_header = self._detect_header(region, line_num)
                region.text_type = self._classify_text_type(region)
                region.font_size = self._estimate_font_size(region)
                
                formatted_regions.append(region)
        
        logger.info(f"Organized into {len(lines)} lines with structure detection")
        return formatted_regions
    
    def _spatial_sort(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Sort regions by spatial position (reading order)"""
        def sort_key(region):
            bbox = region.bbox
            # Calculate center point
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Primary sort: top to bottom (Y coordinate)
            # Secondary sort: left to right (X coordinate)
            return (center_y, center_x)
        
        return sorted(regions, key=sort_key)
    
    def _group_into_lines(self, regions: List[TextRegion], image_height: int) -> List[List[TextRegion]]:
        """Group regions into text lines with improved algorithm"""
        if not regions:
            return []
        
        lines = []
        current_line = [regions[0]]
        
        for region in regions[1:]:
            # Calculate Y position and height for current region
            current_y_coords = [point[1] for point in region.bbox]
            current_y = sum(current_y_coords) / len(current_y_coords)
            current_height = max(current_y_coords) - min(current_y_coords)
            
            # Calculate Y position and average height for current line
            line_y_coords = []
            line_heights = []
            for r in current_line:
                y_coords = [point[1] for point in r.bbox]
                line_y_coords.extend(y_coords)
                line_heights.append(max(y_coords) - min(y_coords))
            
            line_y = sum(line_y_coords) / len(line_y_coords)
            avg_line_height = sum(line_heights) / len(line_heights)
            
            # Adaptive threshold based on text height
            threshold = max(avg_line_height * 0.6, current_height * 0.6, 15)
            
            # Check if region belongs to current line
            if abs(current_y - line_y) < threshold:
                current_line.append(region)
            else:
                # Finalize current line and start new one
                current_line.sort(key=lambda r: sum(point[0] for point in r.bbox) / len(r.bbox))
                lines.append(current_line)
                current_line = [region]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda r: sum(point[0] for point in r.bbox) / len(r.bbox))
            lines.append(current_line)
        
        return lines
    
    def _detect_title(self, region: TextRegion, line_num: int, total_lines: int) -> bool:
        """Advanced title detection"""
        text = region.text.strip()
        
        # Check position (early lines more likely to be titles)
        position_score = max(0, (5 - line_num) / 5) if line_num < 5 else 0
        
        # Check text patterns
        pattern_score = 0
        for pattern in self.title_patterns:
            if re.match(pattern, text):
                pattern_score = 1
                break
        
        # Check length (titles are usually shorter)
        length_score = max(0, (50 - len(text)) / 50) if len(text) <= 50 else 0
        
        # Check confidence (titles often have high confidence)
        confidence_score = min(1, region.confidence)
        
        # Combined scoring
        total_score = (position_score * 0.3 + pattern_score * 0.4 + 
                      length_score * 0.2 + confidence_score * 0.1)
        
        return total_score > 0.6
    
    def _detect_header(self, region: TextRegion, line_num: int) -> bool:
        """Detect headers (different from titles)"""
        text = region.text.strip()
        
        # Header patterns
        header_patterns = [
            r'^\d+\.\d+',  # Numbered subsections
            r'^[A-Z][a-z]*:',  # Capitalized word with colon
            r'^\([a-z]\)',  # Lettered items
        ]
        
        return any(re.match(pattern, text) for pattern in header_patterns)
    
    def _classify_text_type(self, region: TextRegion) -> str:
        """Classify text into types"""
        if region.is_title:
            return 'title'
        elif region.is_header:
            return 'header'
        elif len(region.text) < 20:
            return 'caption'
        elif region.text.strip().endswith('.'):
            return 'paragraph'
        else:
            return 'text'
    
    def _estimate_font_size(self, region: TextRegion) -> int:
        """Estimate font size based on bounding box height"""
        try:
            y_coords = [point[1] for point in region.bbox]
            height = max(y_coords) - min(y_coords)
            # Rough estimation: bbox height to font size
            estimated_size = max(8, min(72, int(height * 0.8)))
            return estimated_size
        except:
            return None
    
    def format_output(self, regions: List[TextRegion]) -> Dict[str, str]:
        """Format regions into multiple output formats"""
        if not regions:
            return {
                "plain_text": "",
                "formatted_text": "",
                "markdown_text": "",
                "structured_data": "[]",
                "statistics": "{}"
            }
        
        # Group by lines
        lines = {}
        for region in regions:
            line_num = region.line_number
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(region)
        
        # Build different output formats
        plain_lines = []
        formatted_lines = []
        markdown_lines = []
        structured_data = []
        
        # Statistics
        stats = {
            "total_regions": len(regions),
            "total_lines": len(lines),
            "average_confidence": float(np.mean([r.confidence for r in regions])),
            "languages_detected": list(set(r.language for r in regions if r.language != 'unknown')),
            "text_types": {},
            "confidence_distribution": {
                "min": float(min(r.confidence for r in regions)),
                "max": float(max(r.confidence for r in regions)),
                "median": float(np.median([r.confidence for r in regions]))
            }
        }
        
        # Count text types
        for region in regions:
            text_type = region.text_type
            stats["text_types"][text_type] = stats["text_types"].get(text_type, 0) + 1
        
        # Process each line
        for line_num in sorted(lines.keys()):
            line_regions = lines[line_num]
            
            # Combine text from regions in the line
            line_texts = [r.text for r in line_regions]
            line_text = ' '.join(line_texts).strip()
            
            if line_text:
                plain_lines.append(line_text)
                
                # Check formatting
                has_title = any(r.is_title for r in line_regions)
                has_header = any(r.is_header for r in line_regions)
                
                if has_title:
                    # Title formatting
                    formatted_lines.append(f"\n{'=' * len(line_text)}")
                    formatted_lines.append(line_text.upper())
                    formatted_lines.append(f"{'=' * len(line_text)}\n")
                    
                    # Markdown title
                    markdown_lines.append(f"# {line_text}")
                    
                elif has_header:
                    # Header formatting
                    formatted_lines.append(f"\n{line_text}")
                    formatted_lines.append(f"{'-' * len(line_text)}")
                    
                    # Markdown header
                    markdown_lines.append(f"## {line_text}")
                    
                else:
                    # Regular text
                    formatted_lines.append(line_text)
                    markdown_lines.append(line_text)
                
                # Add to structured data
                for region in line_regions:
                    structured_data.append(region.to_dict())
        
        return {
            "plain_text": '\n'.join(plain_lines),
            "formatted_text": '\n'.join(formatted_lines),
            "markdown_text": '\n\n'.join(markdown_lines),
            "structured_data": json.dumps(structured_data, indent=2, ensure_ascii=False),
            "statistics": json.dumps(stats, indent=2, ensure_ascii=False)
        }

class PolyOCRPipeline:
    """
    Complete OCR pipeline with EasyOCR accuracy and advanced features
    """
    
    def __init__(self, languages: List[str] = ['en'], gpu: Optional[bool] = None):
        self.languages = languages
        self.gpu = gpu
        self.ocr_wrapper = None
        self.formatter = TextFormatter()
        
        logger.info("Initialized PolyOCR Pipeline")
        logger.info(f"Languages: {languages}")
        logger.info(f"GPU: {'Auto-detect' if gpu is None else gpu}")
    
    def process_image(self, 
                     image_path: str,
                     output_dir: str = "ocr_output",
                     confidence_threshold: float = 0.25,
                     enhance_image: bool = False,
                     draw_boxes: bool = False,
                     paragraph_mode: bool = False,
                     width_ths: float = 0.7,
                     height_ths: float = 0.7) -> Dict[str, str]:
        """
        Process image with comprehensive OCR pipeline
        """
        try:
            start_time = time.time()
            logger.info(f"Starting OCR processing: {image_path}")
            
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Optional image enhancement
            processing_image_path = image_path
            if enhance_image:
                logger.info("Enhancing image quality...")
                enhanced_path = os.path.join(output_dir, f"{Path(image_path).stem}_enhanced.jpg")
                processing_image_path = ImagePreprocessor.enhance_image(image_path, enhanced_path)
            
            # Initialize OCR wrapper if not already done
            if self.ocr_wrapper is None:
                logger.info("Initializing EasyOCR...")
                self.ocr_wrapper = EasyOCRWrapper(self.languages, self.gpu)
            
            # Extract text regions
            logger.info("Extracting text regions...")
            regions = self.ocr_wrapper.extract_text_regions(
                processing_image_path,
                confidence_threshold=confidence_threshold,
                paragraph_mode=paragraph_mode,
                width_ths=width_ths,
                height_ths=height_ths
            )
            
            if not regions:
                logger.warning("No text detected in image")
                empty_result = {
                    "plain_text": "",
                    "formatted_text": "",
                    "markdown_text": "",
                    "structured_data": "[]",
                    "statistics": json.dumps({"total_regions": 0, "total_lines": 0}, indent=2)
                }
                self._save_results(empty_result, image_path, [], output_dir)
                return empty_result
            
            # Get image dimensions
            img = cv2.imread(processing_image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {processing_image_path}")
            height, width = img.shape[:2]
            
            # Format and organize regions
            logger.info("Organizing and formatting text...")
            organized_regions = self.formatter.organize_regions(regions, width, height)
            output_dict = self.formatter.format_output(organized_regions)
            
            # Save results
            self._save_results(output_dict, image_path, organized_regions, output_dir)
            
            # Draw bounding boxes if requested
            if draw_boxes:
                self._draw_boxes(processing_image_path, organized_regions, output_dir)
            
            # Processing summary
            processing_time = time.time() - start_time
            logger.info(f"OCR processing completed successfully in {processing_time:.2f}s")
            logger.info(f"Processed {len(organized_regions)} text regions across {len(set(r.line_number for r in organized_regions))} lines")
            
            return output_dict
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def process_batch(self, 
                     image_paths: List[str],
                     output_dir: str = "ocr_output_batch",
                     **kwargs) -> Dict[str, Dict[str, str]]:
        """Process multiple images in batch"""
        results = {}
        
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")
                
                # Create individual output directory
                image_output_dir = os.path.join(output_dir, Path(image_path).stem)
                
                result = self.process_image(image_path, image_output_dir, **kwargs)
                results[image_path] = result
                
                logger.info(f"Completed {i}/{len(image_paths)} images")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[image_path] = {"error": str(e)}
        
        # Save batch summary
        batch_summary = {
            "total_images": len(image_paths),
            "successful": len([r for r in results.values() if "error" not in r]),
            "failed": len([r for r in results.values() if "error" in r]),
            "results": results
        }
        
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed. Summary saved: {summary_path}")
        return results
    
    def _save_results(self, output_dict: Dict[str, str], image_path: str,
                     regions: List[TextRegion], output_dir: str):
        """Save comprehensive results to files"""
        base_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save plain text
            plain_path = os.path.join(output_dir, f"{base_name}_plain.txt")
            with open(plain_path, 'w', encoding='utf-8') as f:
                f.write(output_dict['plain_text'])
            logger.info(f"Plain text saved: {plain_path}")
            
            # Save formatted text
            formatted_path = os.path.join(output_dir, f"{base_name}_formatted.txt")
            with open(formatted_path, 'w', encoding='utf-8') as f:
                f.write(output_dict['formatted_text'])
            logger.info(f"Formatted text saved: {formatted_path}")
            
            # Save markdown
            if 'markdown_text' in output_dict:
                markdown_path = os.path.join(output_dir, f"{base_name}_markdown.md")
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(f"# OCR Results for {base_name}\n\n")
                    f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write(output_dict['markdown_text'])
                logger.info(f"Markdown saved: {markdown_path}")
            
            # Save structured data
            structured_path = os.path.join(output_dir, f"{base_name}_structured.json")
            with open(structured_path, 'w', encoding='utf-8') as f:
                f.write(output_dict['structured_data'])
            logger.info(f"Structured data saved: {structured_path}")
            
            # Save statistics
            if 'statistics' in output_dict:
                stats_path = os.path.join(output_dir, f"{base_name}_statistics.json")
                with open(stats_path, 'w', encoding='utf-8') as f:
                    f.write(output_dict['statistics'])
                logger.info(f"Statistics saved: {stats_path}")
            
            # Save comprehensive summary
            summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"PolyOCR Processing Summary\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Regions detected: {len(regions)}\n")
                
                if regions:
                    confidences = [r.confidence for r in regions]
                    f.write(f"Average confidence: {np.mean(confidences):.3f}\n")
                    f.write(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}\n")
                    
                    # Language distribution
                    languages = [r.language for r in regions if r.language != 'unknown']
                    if languages:
                        lang_counts = {}
                        for lang in languages:
                            lang_counts[lang] = lang_counts.get(lang, 0) + 1
                        f.write(f"Languages detected: {dict(lang_counts)}\n")
                    
                    # Text type distribution
                    text_types = [r.text_type for r in regions]
                    type_counts = {}
                    for text_type in text_types:
                        type_counts[text_type] = type_counts.get(text_type, 0) + 1
                    f.write(f"Text types: {dict(type_counts)}\n")
                    
                    # Line count
                    unique_lines = len(set(r.line_number for r in regions))
                    f.write(f"Lines of text: {unique_lines}\n")
                
                f.write(f"\nOutput files generated:\n")
                f.write(f"- Plain text: {base_name}_plain.txt\n")
                f.write(f"- Formatted text: {base_name}_formatted.txt\n")
                f.write(f"- Markdown: {base_name}_markdown.md\n")
                f.write(f"- Structured data: {base_name}_structured.json\n")
                f.write(f"- Statistics: {base_name}_statistics.json\n")
                
            logger.info(f"Summary saved: {summary_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save some results: {e}")
    
    def _draw_boxes(self, image_path: str, regions: List[TextRegion], output_dir: str):
        """Draw bounding boxes with enhanced visualization"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Cannot read image for box drawing: {image_path}")
                return
            
            # Create a copy for drawing
            viz_img = img.copy()
            
            # Color scheme for different text types
            colors = {
                'title': (0, 0, 255),      # Red
                'header': (255, 0, 0),     # Blue
                'paragraph': (0, 255, 0),  # Green
                'caption': (255, 255, 0),  # Cyan
                'text': (128, 128, 128)    # Gray
            }
            
            for i, region in enumerate(regions):
                # Convert bbox to numpy array
                bbox = np.array(region.bbox, dtype=np.int32)
                
                # Get color based on text type
                color = colors.get(region.text_type, (0, 255, 0))
                
                # Draw bounding box
                cv2.polylines(viz_img, [bbox], True, color, 2)
                
                # Add confidence score
                conf_text = f"{region.confidence:.2f}"
                cv2.putText(viz_img, conf_text, 
                           tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
                
                # Add region number
                cv2.putText(viz_img, str(i), 
                           (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, color, 1)
            
            # Add legend
            legend_y = 30
            for text_type, color in colors.items():
                cv2.putText(viz_img, f"{text_type}: ", 
                           (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
                legend_y += 25
            
            # Save visualization
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(output_path, viz_img)
            logger.info(f"Visualization saved: {output_path}")
            
            # Also save a clean version with just boxes
            clean_img = img.copy()
            for region in regions:
                bbox = np.array(region.bbox, dtype=np.int32)
                cv2.polylines(clean_img, [bbox], True, (0, 255, 0), 2)
            
            clean_output_path = os.path.join(output_dir, f"{base_name}_boxes.jpg")
            cv2.imwrite(clean_output_path, clean_img)
            logger.info(f"Clean boxes image saved: {clean_output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to draw bounding boxes: {e}")

def create_config_file(config_path: str = "polyocr_config.json"):
    """Create a configuration file with default settings"""
    config = {
        "default_settings": {
            "languages": ["en"],
            "confidence_threshold": 0.25,
            "gpu": None,
            "output_dir": "ocr_output",
            "enhance_image": False,
            "draw_boxes": False,
            "paragraph_mode": False,
            "width_ths": 0.7,
            "height_ths": 0.7
        },
        "language_codes": {
            "english": "en",
            "hindi": "hi",
            "chinese_simplified": "ch_sim",
            "chinese_traditional": "ch_tra",
            "japanese": "ja",
            "korean": "ko",
            "arabic": "ar",
            "thai": "th",
            "vietnamese": "vi",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "portuguese": "pt",
            "russian": "ru",
            "italian": "it"
        },
        "advanced_settings": {
            "log_level": "INFO",
            "log_file": "polyocr.log",
            "batch_size": 1,
            "max_workers": 4,
            "timeout_seconds": 300
        },
        "output_formats": {
            "save_plain_text": True,
            "save_formatted_text": True,
            "save_markdown": True,
            "save_structured_json": True,
            "save_statistics": True,
            "save_summary": True
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration file created: {config_path}")
    return config_path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config file: {e}")
        return {}

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="PolyOCR Pipeline - Complete OCR solution with EasyOCR accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python polyocr.py image.jpg
    
    # Multi-language processing
    python polyocr.py image.jpg --lang "en+hi+zh"
    
    # Batch processing
    python polyocr.py *.jpg --batch
    
    # With enhancements
    python polyocr.py image.jpg --enhance --draw_boxes --confidence 0.3
    
    # Generate config file
    python polyocr.py --create_config
        """
    )
    
    # Main arguments
    parser.add_argument("image_paths", nargs='*', help="Path(s) to input image(s)")
    parser.add_argument("--output_dir", default="ocr_output", 
                       help="Output directory (default: ocr_output)")
    parser.add_argument("--lang", "--languages", default="en", 
                       help="Languages (e.g., 'en', 'hi', 'zh', 'en+hi+zh')")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold 0.0-1.0 (default: 0.25)")
    
    # Processing options
    parser.add_argument("--gpu", action="store_true", 
                       help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", 
                       help="Force CPU usage")
    parser.add_argument("--enhance", action="store_true",
                       help="Enhance image quality before OCR")
    parser.add_argument("--draw_boxes", action="store_true",
                       help="Draw bounding boxes on image")
    parser.add_argument("--paragraph", action="store_true",
                       help="Enable paragraph mode")
    
    # EasyOCR parameters
    parser.add_argument("--width_ths", type=float, default=0.7,
                       help="Width threshold for text detection (default: 0.7)")
    parser.add_argument("--height_ths", type=float, default=0.7,
                       help="Height threshold for text detection (default: 0.7)")
    
    # Batch processing
    parser.add_argument("--batch", action="store_true",
                       help="Process multiple images in batch mode")
    
    # Configuration
    parser.add_argument("--config", help="Load settings from config file")
    parser.add_argument("--create_config", action="store_true",
                       help="Create default configuration file")
    
    # Logging
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress console output")
    
    # Legacy compatibility (ignored)
    parser.add_argument("--denoising", help=argparse.SUPPRESS)
    parser.add_argument("--craft_model_path", help=argparse.SUPPRESS)
    parser.add_argument("--preserve_formatting", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable_postprocessing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug_crops_dir", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        create_config_file()
        return 0
    
    # Validate input
    if not args.image_paths:
        parser.error("No image paths provided")
    
    try:
        # Setup logging
        if not args.quiet:
            global logger
            logger = setup_logging(log_level=args.log_level)
        
        # Load configuration if specified
        config = {}
        if args.config:
            config = load_config(args.config)
        
        # Parse languages
        languages = [lang.strip() for lang in args.lang.replace('+', ' ').split()]
        
        # Determine GPU usage
        gpu = None
        if args.gpu:
            gpu = True
        elif args.cpu:
            gpu = False
        
        # Expand wildcards for batch processing
        import glob
        image_paths = []
        for path_pattern in args.image_paths:
            if '*' in path_pattern or '?' in path_pattern:
                image_paths.extend(glob.glob(path_pattern))
            else:
                image_paths.append(path_pattern)
        
        # Remove duplicates and validate
        image_paths = list(set(image_paths))
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_paths = [p for p in image_paths if Path(p).suffix.lower() in valid_extensions]
        
        if not image_paths:
            logger.error("No valid image files found")
            return 1
        
        logger.info(f"Found {len(image_paths)} image(s) to process")
        
        # Initialize pipeline
        logger.info("Initializing PolyOCR Pipeline...")
        pipeline = PolyOCRPipeline(languages=languages, gpu=gpu)
        
        # Process images
        if args.batch or len(image_paths) > 1:
            # Batch processing
            logger.info(f"Starting batch processing of {len(image_paths)} images")
            results = pipeline.process_batch(
                image_paths=image_paths,
                output_dir=args.output_dir,
                confidence_threshold=args.confidence,
                enhance_image=args.enhance,
                draw_boxes=args.draw_boxes,
                paragraph_mode=args.paragraph,
                width_ths=args.width_ths,
                height_ths=args.height_ths
            )
            
            # Print batch summary
            successful = len([r for r in results.values() if "error" not in r])
            failed = len(results) - successful
            
            print(f"\n{'=' * 60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total images: {len(image_paths)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Results saved to: {args.output_dir}")
            print(f"{'=' * 60}")
            
        else:
            # Single image processing
            image_path = image_paths[0]
            logger.info(f"Processing single image: {image_path}")
            
            result = pipeline.process_image(
                image_path=image_path,
                output_dir=args.output_dir,
                confidence_threshold=args.confidence,
                enhance_image=args.enhance,
                draw_boxes=args.draw_boxes,
                paragraph_mode=args.paragraph,
                width_ths=args.width_ths,
                height_ths=args.height_ths
            )
            
            # Print results
            print(f"\n{'=' * 60}")
            print("OCR RESULTS")
            print(f"{'=' * 60}")
            if result['formatted_text'].strip():
                print(result['formatted_text'])
            else:
                print("No text detected in image")
            print(f"\n{'=' * 60}")
            print(f"Results saved to: {args.output_dir}")
            print(f"{'=' * 60}")
        
        logger.info("PolyOCR processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.log_level == 'DEBUG':
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())