import argparse
import time, os, sys
import numpy as np
import cv2


from .utils.video import open_video, writer, resize_keep_aspect
from .utils.draw import draw_dets
from .utils.metrics import MetricAggregator
from .utils.denoise import Denoiser

def build_model(name: str, device: str, conf: float, half: bool):
    name = name.lower()
    if name == "yolo":
        from .models.yolo import YOLOv8Detector
        return YOLOv8Detector(device=device, conf=conf, half=half)
    elif name == "detr":
        from .models.detr import DETRDetector
        return DETRDetector(device=device, conf=conf)
    else:
        raise ValueError(f"Unknown model: {name} (expected 'yolo' or 'detr')")

def main():
    ap = argparse.ArgumentParser(description="People detection on video with two alternative models (YOLOv8 / DETR).")
    ap.add_argument("--model", type=str, required=True, choices=["yolo", "detr"], help="Model to run.")
    ap.add_argument("--input", type=str, default="crowd.mp4", help="Path to input video.")
    ap.add_argument("--output", type=str, default="out.mp4", help="Path to output annotated video.")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0|...")
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold.")
    ap.add_argument("--half", action="store_true", help="Use FP16 (YOLO only, CUDA required).")
    ap.add_argument("--resize", type=int, default=0, help="Long-side resize (e.g., 1280). 0 = keep native.")
    ap.add_argument("--denoise", type=str, default="none",
                    choices=["none", "gauss", "median", "bilateral", "fastnlm", "fastnlm_multi"],
                    help="Denoising method.")
    ap.add_argument("--denoise-ksize", type=int, default=5, help="Kernel size for gauss/median/bilateral.")
    ap.add_argument("--denoise-sigma", type=float, default=0.0, help="Gaussian sigma.")
    ap.add_argument("--denoise-bilat-sigma-color", type=float, default=50.0, help="Bilateral sigmaColor.")
    ap.add_argument("--denoise-bilat-sigma-space", type=float, default=7.0, help="Bilateral sigmaSpace.")
    ap.add_argument("--denoise-nlm-h", type=float, default=7.0, help="FastNLM strength (luma).")
    ap.add_argument("--denoise-nlm-h-color", type=float, default=7.0, help="FastNLM strength (chroma).")
    ap.add_argument("--denoise-nlm-template", type=int, default=7, help="FastNLM template window.")
    ap.add_argument("--denoise-nlm-search", type=int, default=21, help="FastNLM search window.")
    ap.add_argument("--denoise-multi-window", type=int, default=5, help="Temporal window for fastnlm_multi.")

    args = ap.parse_args()

    det = build_model(args.model, args.device, args.conf, args.half)
    cap, w0, h0, fps0 = open_video(args.input)

    # Decide output size
    frame0_ok, frame0 = cap.read()
    if not frame0_ok:
        raise RuntimeError("Empty/invalid video.")
    if args.resize > 0:
        frame0 = resize_keep_aspect(frame0, args.resize)
    oh, ow = frame0.shape[:2]

    out = writer(args.output, ow, oh, fps0)
    det.warmup(frame0)

    denoiser = Denoiser(
        method=args.denoise,
        ksize=args.denoise_ksize,
        sigma=args.denoise_sigma,
        bilateral_sigma_color=args.denoise_bilat_sigma_color,
        bilateral_sigma_space=args.denoise_bilat_sigma_space,
        nlm_h=args.denoise_nlm_h,
        nlm_h_color=args.denoise_nlm_h_color,
        nlm_template=args.denoise_nlm_template,
        nlm_search=args.denoise_nlm_search,
        multi_window=args.denoise_multi_window,
    )

    metrics = MetricAggregator()
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.resize > 0:
            frame = resize_keep_aspect(frame, args.resize)

        # optional denoising pre-processing
        frame = denoiser.apply(frame)

        t0 = time.perf_counter()
        boxes, scores, _ = det.infer(frame)

        t1 = time.perf_counter()
        metrics.update(boxes, t0, t1)

        vis = draw_dets(frame, boxes, scores, title=f"{det.name()} | conf>={args.conf:.2f}")
        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()

    # Save metrics
    base = os.path.splitext(args.output)[0]
    mjson = f"{base.replace('.mp4','')}_metrics_{args.model}.json"
    mcsv  = f"{base.replace('.mp4','')}_metrics_{args.model}.csv"
    metrics.save(mjson, mcsv)

    # Print summary
    print(f"Saved video: {args.output}")
    print(f"Saved metrics: {mjson}, {mcsv}")
    print("Summary:", metrics.summary())

if __name__ == "__main__":
    main()
