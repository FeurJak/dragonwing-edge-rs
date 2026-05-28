//! YOLO detection post-processing.
//!
//! This module provides utilities for decoding YOLOv8 detection outputs
//! and applying Non-Maximum Suppression (NMS).
//!
//! YOLOv8 is anchor-free, outputting:
//! - `[batch, num_detections, 4 + num_classes]`
//! - Where 4 = `(cx, cy, w, h)` in relative coordinates (0-1)

/// A single detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in pixel coordinates: [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Class ID (0-indexed).
    pub class_id: u32,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
}

impl Detection {
    /// Create a new detection.
    pub fn new(bbox: [f32; 4], class_id: u32, confidence: f32) -> Self {
        Self { bbox, class_id, confidence }
    }

    /// Calculate the area of the bounding box.
    pub fn area(&self) -> f32 {
        let w = (self.bbox[2] - self.bbox[0]).max(0.0);
        let h = (self.bbox[3] - self.bbox[1]).max(0.0);
        w * h
    }
}

/// Calculate Intersection over Union (IoU) between two detections.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.bbox[0].max(b.bbox[0]);
    let y1 = a.bbox[1].max(b.bbox[1]);
    let x2 = a.bbox[2].min(b.bbox[2]);
    let y2 = a.bbox[3].min(b.bbox[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = a.area();
    let area_b = b.area();
    let union_area = area_a + area_b - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

/// Decode YOLOv8 raw output to detections.
///
/// YOLOv8n output shape: `[1, num_classes + 4, num_detections]` or 
/// `[1, num_detections, num_classes + 4]` depending on export settings.
///
/// The output format for opset 12 export with `simplify=True` is typically:
/// - Shape: `[1, 84, 8400]` for COCO (80 classes + 4 bbox coords)
/// - Layout: `[batch, cx/cy/w/h + class_probs, anchors]`
///
/// # Arguments
///
/// * `raw_output` - The raw model output tensor
/// * `num_classes` - Number of detection classes (e.g., 80 for COCO)
/// * `num_detections` - Number of detection anchors (e.g., 8400)
/// * `img_width` - Original image width for scaling
/// * `img_height` - Original image height for scaling
/// * `conf_threshold` - Minimum confidence threshold
///
/// # Returns
///
/// A vector of `Detection` objects with confidence above the threshold.
pub fn decode_detections_v8(
    raw_output: &[f32],
    num_classes: usize,
    num_detections: usize,
    img_width: u32,
    img_height: u32,
    conf_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    
    // YOLOv8 output is [1, 84, 8400] for COCO
    // Rows 0-3: cx, cy, w, h (relative to model input size, e.g., 640x640)
    // Rows 4-83: class probabilities
    
    let expected_len = (4 + num_classes) * num_detections;
    if raw_output.len() < expected_len {
        return detections;
    }
    
    // Model input size (typically 640x640 for YOLOv8)
    let model_size = 640.0f32;
    let scale_x = img_width as f32 / model_size;
    let scale_y = img_height as f32 / model_size;
    
    for anchor in 0..num_detections {
        // Extract bbox coordinates (row-major: [channel, anchor])
        let cx = raw_output[0 * num_detections + anchor];
        let cy = raw_output[1 * num_detections + anchor];
        let w = raw_output[2 * num_detections + anchor];
        let h = raw_output[3 * num_detections + anchor];
        
        // Find best class
        let mut best_class = 0;
        let mut best_conf = 0.0f32;
        
        for cls in 0..num_classes {
            let conf = raw_output[(4 + cls) * num_detections + anchor];
            if conf > best_conf {
                best_conf = conf;
                best_class = cls;
            }
        }
        
        // Skip low-confidence detections
        if best_conf < conf_threshold {
            continue;
        }
        
        // Convert center+size to corner coordinates
        let x1 = (cx - w / 2.0) * scale_x;
        let y1 = (cy - h / 2.0) * scale_y;
        let x2 = (cx + w / 2.0) * scale_x;
        let y2 = (cy + h / 2.0) * scale_y;
        
        // Clamp to image bounds
        let x1 = x1.max(0.0).min(img_width as f32);
        let y1 = y1.max(0.0).min(img_height as f32);
        let x2 = x2.max(0.0).min(img_width as f32);
        let y2 = y2.max(0.0).min(img_height as f32);
        
        detections.push(Detection::new(
            [x1, y1, x2, y2],
            best_class as u32,
            best_conf,
        ));
    }
    
    detections
}

/// Decode YOLOv8 raw output (alternative layout: [1, num_detections, 84]).
///
/// Some exports use `[1, 8400, 84]` layout instead of `[1, 84, 8400]`.
pub fn decode_detections_v8_alt(
    raw_output: &[f32],
    num_classes: usize,
    num_detections: usize,
    img_width: u32,
    img_height: u32,
    conf_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    
    let row_size = 4 + num_classes;
    let expected_len = num_detections * row_size;
    if raw_output.len() < expected_len {
        return detections;
    }
    
    let model_size = 640.0f32;
    let scale_x = img_width as f32 / model_size;
    let scale_y = img_height as f32 / model_size;
    
    for anchor in 0..num_detections {
        let offset = anchor * row_size;
        
        let cx = raw_output[offset + 0];
        let cy = raw_output[offset + 1];
        let w = raw_output[offset + 2];
        let h = raw_output[offset + 3];
        
        // Find best class
        let mut best_class = 0;
        let mut best_conf = 0.0f32;
        
        for cls in 0..num_classes {
            let conf = raw_output[offset + 4 + cls];
            if conf > best_conf {
                best_conf = conf;
                best_class = cls;
            }
        }
        
        if best_conf < conf_threshold {
            continue;
        }
        
        let x1 = ((cx - w / 2.0) * scale_x).max(0.0).min(img_width as f32);
        let y1 = ((cy - h / 2.0) * scale_y).max(0.0).min(img_height as f32);
        let x2 = ((cx + w / 2.0) * scale_x).max(0.0).min(img_width as f32);
        let y2 = ((cy + h / 2.0) * scale_y).max(0.0).min(img_height as f32);
        
        detections.push(Detection::new(
            [x1, y1, x2, y2],
            best_class as u32,
            best_conf,
        ));
    }
    
    detections
}

/// Apply Non-Maximum Suppression (NMS) to a list of detections.
///
/// NMS removes overlapping detections for the same class, keeping only
/// the highest-confidence one from each cluster.
///
/// # Arguments
///
/// * `detections` - Input detections (will be sorted by confidence)
/// * `iou_threshold` - Maximum IoU allowed between kept detections
///
/// # Algorithm
///
/// 1. Sort detections by confidence (descending)
/// 2. For each detection (starting with highest confidence):
///    - If not suppressed, add to output
///    - Suppress all lower-confidence detections of the same class
///      with IoU > threshold
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    // Sort by confidence descending
    detections.sort_by(|a, b| {
        b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let mut keep = vec![true; detections.len()];
    
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            
            // Only suppress same-class detections
            if detections[i].class_id != detections[j].class_id {
                continue;
            }
            
            // Suppress if IoU > threshold
            if iou(&detections[i], &detections[j]) > iou_threshold {
                keep[j] = false;
            }
        }
    }
    
    // Remove suppressed detections (iterate in reverse to preserve indices)
    let mut i = detections.len();
    while i > 0 {
        i -= 1;
        if !keep[i] {
            detections.swap_remove(i);
        }
    }
}

/// Apply class-agnostic NMS (for when classes overlap naturally).
///
/// This variant suppresses overlapping detections regardless of class.
pub fn nms_agnostic(detections: &mut Vec<Detection>, iou_threshold: f32) {
    detections.sort_by(|a, b| {
        b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let mut keep = vec![true; detections.len()];
    
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            
            if iou(&detections[i], &detections[j]) > iou_threshold {
                keep[j] = false;
            }
        }
    }
    
    let mut i = detections.len();
    while i > 0 {
        i -= 1;
        if !keep[i] {
            detections.swap_remove(i);
        }
    }
}

/// Full YOLO post-processing pipeline.
///
/// This convenience function combines decoding and NMS into a single call.
///
/// # Arguments
///
/// * `raw_output` - Raw model output tensor
/// * `num_classes` - Number of classes (80 for COCO)
/// * `num_detections` - Number of anchors (8400 for YOLOv8n at 640x640)
/// * `img_width` - Image width for coordinate scaling
/// * `img_height` - Image height for coordinate scaling
/// * `conf_threshold` - Minimum confidence (typically 0.25)
/// * `iou_threshold` - NMS IoU threshold (typically 0.45)
///
/// # Returns
///
/// Filtered and NMS'd detections ready for display/use.
pub fn postprocess_yolo(
    raw_output: &[f32],
    num_classes: usize,
    num_detections: usize,
    img_width: u32,
    img_height: u32,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Vec<Detection> {
    let mut detections = decode_detections_v8(
        raw_output,
        num_classes,
        num_detections,
        img_width,
        img_height,
        conf_threshold,
    );
    
    nms(&mut detections, iou_threshold);
    
    detections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_no_overlap() {
        let a = Detection::new([0.0, 0.0, 10.0, 10.0], 0, 0.9);
        let b = Detection::new([20.0, 20.0, 30.0, 30.0], 0, 0.8);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn test_iou_full_overlap() {
        let a = Detection::new([0.0, 0.0, 10.0, 10.0], 0, 0.9);
        let b = Detection::new([0.0, 0.0, 10.0, 10.0], 0, 0.8);
        assert!((iou(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = Detection::new([0.0, 0.0, 10.0, 10.0], 0, 0.9);
        let b = Detection::new([5.0, 0.0, 15.0, 10.0], 0, 0.8);
        // Intersection: 5x10 = 50, Union: 100 + 100 - 50 = 150
        let expected_iou = 50.0 / 150.0;
        assert!((iou(&a, &b) - expected_iou).abs() < 1e-5);
    }

    #[test]
    fn test_nms_basic() {
        let mut detections = vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0, 0.9),
            Detection::new([1.0, 1.0, 11.0, 11.0], 0, 0.8), // Overlaps with first, same class
            Detection::new([20.0, 20.0, 30.0, 30.0], 0, 0.7), // No overlap
        ];
        
        nms(&mut detections, 0.5);
        
        // Should keep the high-conf one and the non-overlapping one
        assert_eq!(detections.len(), 2);
        assert!((detections[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_nms_different_classes() {
        let mut detections = vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0, 0.9),
            Detection::new([0.0, 0.0, 10.0, 10.0], 1, 0.8), // Same box, different class
        ];
        
        nms(&mut detections, 0.5);
        
        // Different classes should both be kept
        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn test_decode_empty() {
        let detections = decode_detections_v8(&[], 80, 8400, 640, 480, 0.25);
        assert!(detections.is_empty());
    }

    #[test]
    fn test_detection_area() {
        let det = Detection::new([0.0, 0.0, 10.0, 20.0], 0, 0.9);
        assert!((det.area() - 200.0).abs() < 1e-5);
    }
}
