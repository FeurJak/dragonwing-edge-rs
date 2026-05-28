# YOLO Pipeline

This document describes the YOLOv8 detection pipeline implementation in dragonwing.

## Architecture Overview

YOLOv8 is an anchor-free object detector with three main components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          YOLOv8n Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input (640×640×3)                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────┐                    │
│  │           Backbone (CSPDarknet53)           │                    │
│  │  Conv → SiLU → C2f blocks → Downsample      │                    │
│  │  Feature maps at 1/8, 1/16, 1/32 scales     │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────┐                    │
│  │              Neck (FPN + PAN)               │                    │
│  │  Upsample → Concat → Conv → Concat          │                    │
│  │  Multi-scale feature fusion                 │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────┐                    │
│  │          Detection Heads (×3)               │                    │
│  │  Per-scale: Conv → Sigmoid → Output         │                    │
│  │  Output: [1, 84, 8400] (80 classes + bbox)  │                    │
│  └─────────────────────────────────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Backbone: CSPDarknet

The backbone extracts features at multiple scales:

- **Stage 1**: 640 → 320 (stride 2)
- **Stage 2**: 320 → 160 (stride 4)
- **Stage 3**: 160 → 80 (stride 8) — P3 output
- **Stage 4**: 80 → 40 (stride 16) — P4 output
- **Stage 5**: 40 → 20 (stride 32) — P5 output

Each stage uses **C2f blocks** (Cross-Stage Partial with 2 convolutions):

```
Input → Split → Conv → Concat → Conv → Output
           ↓              ↑
           └──────────────┘
```

### Neck: FPN + PAN

The neck fuses multi-scale features bidirectionally:

**Top-Down (FPN)**:

```
P5 (20×20) ─┬─ Upsample ─┬─ Concat with P4 ─┬─ C2f ─→ N4
            │            │                  │
P4 (40×40) ─┘            └─ Upsample ─ Concat with P3 ─ C2f ─→ N3
```

**Bottom-Up (PAN)**:

```
N3 (80×80) ─┬─ Downsample ─┬─ Concat with N4 ─┬─ C2f ─→ Head3
            │              │                  │
N4 (40×40) ─┘              └─ Downsample ─ Concat with P5 ─ C2f ─→ Head4/5
```

### Detection Heads

YOLOv8 is **anchor-free**. Each detection head outputs:

| Channel | Content                                   |
| ------- | ----------------------------------------- |
| 0-3     | Bounding box: (cx, cy, w, h)              |
| 4-83    | Class probabilities (80 classes for COCO) |

Output shapes per scale:

- P3: [1, 84, 6400] (80×80 grid = 6400 anchors)
- P4: [1, 84, 1600] (40×40 grid)
- P5: [1, 84, 400] (20×20 grid)

Combined: [1, 84, 8400]

## SiLU Activation

YOLOv8 uses **SiLU** (Sigmoid Linear Unit) throughout:

```
SiLU(x) = x × sigmoid(x)
```

In ONNX, this appears as:

```
x → Sigmoid → y
x, y → Mul → output
```

Our fusion pass detects and optimizes this pattern.

## Post-Processing

### Detection Decoding

Convert raw model output to bounding boxes:

```rust
// crates/dragonwing-onnx/src/postprocess.rs:78
pub fn decode_detections_v8(
    raw_output: &[f32],       // [1, 84, 8400]
    num_classes: usize,       // 80
    num_detections: usize,    // 8400
    img_width: u32,
    img_height: u32,
    conf_threshold: f32,      // e.g., 0.25
) -> Vec<Detection>
```

For each detection anchor:

1. Extract bbox coords: `(cx, cy, w, h)`
2. Find max class probability
3. If max_prob > threshold:
    - Scale bbox to image coordinates
    - Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    - Store Detection

### Non-Maximum Suppression (NMS)

Remove overlapping detections:

```rust
// crates/dragonwing-onnx/src/postprocess.rs:250
pub fn nms(
    detections: &mut Vec<Detection>,
    iou_threshold: f32,  // e.g., 0.45
)
```

Algorithm:

1. Sort detections by confidence (descending)
2. For each detection:
    - If not suppressed, keep it
    - Suppress all lower-confidence detections with IoU > threshold

The implementation is **class-aware**: only detections of the same class compete.

### IoU Calculation

```rust
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let inter = intersection_area(a, b);
    let union = a.area() + b.area() - inter;
    inter / union
}
```

## ONNX Export Settings

To export YOLOv8 for dragonwing:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(
    format='onnx',
    opset=12,        # Match our supported opset
    simplify=True,   # Run onnx-simplifier
    dynamic=False,   # Static shapes only
    imgsz=640,       # Input resolution
)
```

Output file: `yolov8n.onnx` (~13 MB)

## Ops Required by YOLO

| Op        | Count | Notes                  |
| --------- | ----- | ---------------------- |
| Conv      | ~75   | Standard + depthwise   |
| Sigmoid   | ~20   | Part of SiLU           |
| Mul       | ~20   | Part of SiLU + scaling |
| Add       | ~30   | Skip connections       |
| Concat    | ~15   | Feature fusion         |
| Resize    | ~5    | Upsampling in FPN      |
| Split     | ~10   | C2f blocks             |
| MaxPool   | ~5    | SPPF module            |
| Reshape   | ~5    | Output formatting      |
| Transpose | ~3    | Output formatting      |

Total: ~200 ops

## Memory Requirements

For YOLOv8n at 640×640 (F32):

| Component          | Size        |
| ------------------ | ----------- |
| Weights            | ~12 MiB     |
| Activations (peak) | ~80 MiB     |
| Output buffers     | ~2.5 MiB    |
| **Total**          | **~95 MiB** |

With slab allocator buffer reuse, peak can be reduced to ~60 MiB.

## Performance Characteristics

On Arduino UNO Q (Adreno A702 + Cortex-A53):

| Metric       | CPU    | Vulkan (estimated) |
| ------------ | ------ | ------------------ |
| Inference    | ~800ms | ~150ms             |
| Post-process | ~5ms   | ~5ms (CPU)         |
| Total        | ~805ms | ~155ms             |

Note: These are estimates. INT8 quantization (task 006) expected to provide 2-4× speedup.

## Usage Example

```rust
use dragonwing_onnx::{
    load_model, compile_model, CpuGraphRuntime,
    postprocess_yolo, apply_fusion_passes,
};

// Load and compile model
let model = load_model("yolov8n.onnx")?;
let mut graph = compile_model(&model, Dtype::F32)?;

// Apply optimizations
apply_fusion_passes(&mut graph);
convert_nchw_to_nhwc(&mut graph)?;

// Create runtime
let mut runtime = CpuGraphRuntime::new(graph)?;

// Run inference
let input = preprocess_image(&image, 640, 640);
runtime.set_input_f32("images", &input)?;
runtime.run()?;
let output = runtime.get_output_f32("output0")?;

// Post-process
let detections = postprocess_yolo(
    &output,
    80,          // num_classes (COCO)
    8400,        // num_detections
    image.width(),
    image.height(),
    0.25,        // conf_threshold
    0.45,        // iou_threshold
);

for det in &detections {
    println!("Class {} at [{:.0}, {:.0}, {:.0}, {:.0}] conf={:.2}",
        det.class_id,
        det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
        det.confidence
    );
}
```

## File References

- `crates/dragonwing-onnx/src/postprocess.rs:1-408` — Detection decoding and NMS
- `crates/dragonwing-onnx/src/fusion.rs:1-386` — SiLU and Conv-Relu fusion
- `crates/dragonwing-test/src/yolo_e2e.rs:1-352` — Integration tests
