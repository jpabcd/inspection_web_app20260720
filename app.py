import argparse
import glob
import hashlib
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ANNOTATION_DIR = APP_DIR / "annotations"
ANNOTATION_FILE = ANNOTATION_DIR / "annotations.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASSIFICATION_VERDICT_MAP = {
    "OK": "分类正确",
    "NG": "分类错误",
    "分类正确": "分类正确",
    "分类错误": "分类错误",
}


def normalize_path_text(value):
    return str(value).replace("\\", "/").rstrip("/")


def path_basename(value):
    return os.path.basename(normalize_path_text(value)).lower()


def get_model_prediction(image_path):
    stem = Path(image_path).stem
    tokens = [token.upper() for token in stem.split("_")]
    if "OK" in tokens:
        return "合格品"
    if "NG" in tokens:
        return "缺陷品"
    return "未知"


def get_light_type(image_path):
    parts = [part.lower() for part in str(image_path).replace("\\", "/").split("/")]
    for part in parts:
        if re.fullmatch(r"light_\d+", part):
            return part

    stem_tokens = Path(image_path).stem.lower().split("_")
    for index, token in enumerate(stem_tokens[:-1]):
        if token == "light" and stem_tokens[index + 1].isdigit():
            return f"light_{stem_tokens[index + 1]}"
    return "unknown"


def empty_stats():
    return {
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "skipped": 0,
    }


def finalize_stats(stats):
    false_positive_denominator = stats["fp"] + stats["tn"]
    false_negative_denominator = stats["fn"] + stats["tp"]
    return {
        **stats,
        "total": stats["tn"] + stats["fp"] + stats["fn"] + stats["tp"],
        "falsePositiveRate": stats["fp"] / false_positive_denominator if false_positive_denominator else None,
        "falseNegativeRate": stats["fn"] / false_negative_denominator if false_negative_denominator else None,
        "falsePositiveDenominator": false_positive_denominator,
        "falseNegativeDenominator": false_negative_denominator,
    }


def accumulate_stats(stats, path, annotation):
    normalized = normalize_annotation(annotation)
    verdict = normalized.get("verdict", "")
    model_prediction = get_model_prediction(path)

    if verdict not in ("分类正确", "分类错误") or model_prediction not in ("合格品", "缺陷品"):
        stats["skipped"] += 1
        return

    if model_prediction == "合格品" and verdict == "分类正确":
        stats["tn"] += 1
    elif model_prediction == "合格品" and verdict == "分类错误":
        stats["fn"] += 1
    elif model_prediction == "缺陷品" and verdict == "分类错误":
        stats["fp"] += 1
    elif model_prediction == "缺陷品" and verdict == "分类正确":
        stats["tp"] += 1


def get_confusion_cell(path, annotation):
    stats = empty_stats()
    accumulate_stats(stats, path, annotation)
    if stats["skipped"]:
        return ""
    for cell in ("tp", "fn", "fp", "tn"):
        if stats[cell]:
            return cell.upper()
    return ""


def matches_confusion_filter(path, annotations, confusion_cell, confusion_light, annotation_index=None):
    confusion_cell = (confusion_cell or "").upper()
    confusion_light = confusion_light or ""
    if confusion_cell in ("", "ALL"):
        return True
    if confusion_cell not in ("TP", "FN", "FP", "TN"):
        return True
    if confusion_light and confusion_light != "All" and get_light_type(path) != confusion_light:
        return False
    annotation = get_annotation_for_image(path, annotations, annotation_index)
    return get_confusion_cell(path, annotation) == confusion_cell


def load_annotations():
    if not ANNOTATION_FILE.exists():
        return {}
    try:
        with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_annotations(data):
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = ANNOTATION_FILE.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_file.replace(ANNOTATION_FILE)


def normalize_annotation(annotation):
    if not isinstance(annotation, dict):
        annotation = {}

    issues = annotation.get("detectionIssues")
    if not isinstance(issues, list):
        old_type = annotation.get("defectType", "")
        issues = [old_type] if old_type else []

    clean_issues = []
    for issue in issues:
        if issue in ("漏检", "错检") and issue not in clean_issues:
            clean_issues.append(issue)

    miss_regions = annotation.get("missRegions", [])
    if not isinstance(miss_regions, list):
        miss_regions = []

    false_regions = annotation.get("falseRegions", [])
    if not isinstance(false_regions, list):
        false_regions = []

    green_defect_regions = annotation.get("greenDefectRegions", [])
    if not isinstance(green_defect_regions, list):
        green_defect_regions = []

    verdict = CLASSIFICATION_VERDICT_MAP.get(annotation.get("verdict", ""), annotation.get("verdict", ""))
    green_defect = bool(annotation.get("greenDefect", False))
    if annotation.get("logicVerdict") == "NG" or green_defect_regions:
        green_defect = True
    if verdict != "分类错误":
        green_defect = False
        green_defect_regions = []

    if miss_regions and "漏检" not in clean_issues:
        clean_issues.append("漏检")
    if false_regions and "错检" not in clean_issues:
        clean_issues.append("错检")

    return {
        "verdict": verdict,
        "greenDefect": green_defect,
        "greenDefectRegions": green_defect_regions,
        "detectionIssues": clean_issues,
        "defectType": clean_issues[0] if clean_issues else "",
        "missRegions": miss_regions,
        "falseRegions": false_regions,
        "note": annotation.get("note", ""),
        "imageName": annotation.get("imageName", ""),
        "updatedAt": annotation.get("updatedAt", ""),
    }


def annotations_from_import(payload):
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        rows = payload["items"]
    elif isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = [
            {"originalPath": path, **annotation}
            for path, annotation in payload.items()
            if isinstance(annotation, dict)
        ]
    else:
        return {}

    imported = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        original_path = row.get("originalPath", "")
        if not original_path:
            continue
        annotation = normalize_annotation(row)
        annotation["imageName"] = annotation.get("imageName") or path_basename(original_path)
        imported[original_path] = annotation
    return imported


def auto_crop_image(image_path, base_dir, padding=5):
    rel_path = os.path.relpath(image_path, base_dir)
    cache_path = os.path.join(base_dir, "cropped_cache", rel_path)

    if os.path.exists(cache_path):
        return cache_path

    return image_path


def get_filtered_images(light_type, keyword, base_dir):
    keyword = (keyword or "").lower()
    if light_type == "All":
        patterns = [
            os.path.join(base_dir, "*", "yolo_pred_images", f"*{ext}")
            for ext in IMAGE_EXTENSIONS
        ]
    else:
        patterns = [
            os.path.join(base_dir, light_type, "yolo_pred_images", f"*{ext}")
            for ext in IMAGE_EXTENSIONS
        ]

    images = []
    for pattern in patterns:
        images.extend(glob.glob(pattern))

    return sorted(
        img for img in images
        if keyword in os.path.basename(img).lower()
    )


def annotation_matches_image(image_path, annotation_path, annotation):
    image_path_text = normalize_path_text(image_path).lower()
    annotation_path_text = normalize_path_text(annotation_path).lower()
    image_name = path_basename(image_path)
    annotation_name = path_basename(annotation.get("imageName") or annotation_path)
    return image_path_text == annotation_path_text or image_name == annotation_name


def build_annotation_index(annotations):
    indexed = {}
    for annotation_path, annotation in annotations.items():
        annotation_name = path_basename(annotation.get("imageName") or annotation_path)
        if annotation_name and annotation_name not in indexed:
            indexed[annotation_name] = annotation
    return indexed


def get_annotation_for_image(image_path, annotations, annotation_index=None):
    exact = annotations.get(image_path)
    if exact:
        return exact

    if annotation_index is not None:
        indexed = annotation_index.get(path_basename(image_path))
        if indexed:
            return indexed

    for annotation_path, annotation in annotations.items():
        if annotation_matches_image(image_path, annotation_path, annotation):
            return annotation
    return {}


def image_matches_search(image_path, annotations, search_text, annotation_index=None):
    search_text = (search_text or "").strip().lower()
    if not search_text:
        return True

    candidates = [image_path, os.path.basename(image_path)]
    annotation = get_annotation_for_image(image_path, annotations, annotation_index)
    if annotation:
        normalized = normalize_annotation(annotation)
        candidates.extend([
            normalized.get("imageName", ""),
            normalized.get("note", ""),
            normalized.get("verdict", ""),
            "有缺陷但框是绿色的" if normalized.get("greenDefect") else "",
            " ".join(normalized.get("detectionIssues", [])) if isinstance(normalized.get("detectionIssues"), list) else "",
        ])

    return any(search_text in str(candidate).lower() for candidate in candidates)


def order_json_first(paths, annotations, annotation_index=None):
    annotated = []
    plain = []
    for path in paths:
        if get_annotation_for_image(path, annotations, annotation_index):
            annotated.append(path)
        else:
            plain.append(path)
    return annotated + plain


def shuffle_paths(paths, seed):
    seed = seed or "default"
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(f"{seed}|{path}".encode("utf-8")).hexdigest()
    )


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class InspectionHandler(BaseHTTPRequestHandler):
    server_version = "InspectionWeb/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(STATIC_DIR / "index.html")
        elif parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/").lstrip("/")
            self.serve_file(STATIC_DIR / rel)
        elif parsed.path == "/api/images":
            self.handle_images(parse_qs(parsed.query))
        elif parsed.path == "/api/image":
            self.handle_image(parse_qs(parsed.query))
        elif parsed.path == "/api/annotations":
            self.handle_annotations_export()
        elif parsed.path == "/api/stats":
            self.handle_stats()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotation":
            self.handle_save_annotation()
        elif parsed.path == "/api/annotations/bulk-default-correct":
            self.handle_bulk_default_correct()
        elif parsed.path == "/api/annotations/import":
            self.handle_annotations_import()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_file(self, path):
        try:
            resolved = Path(path).resolve()
            if not str(resolved).startswith(str(STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            data = resolved.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json_download(self, payload, filename):
        data = json_bytes(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_images(self, query):
        base_dir = query.get("base_dir", [""])[0].strip()
        light_type = query.get("light_type", ["All"])[0] or "All"
        keyword = query.get("keyword", [""])[0]
        page = max(1, int(float(query.get("page", ["1"])[0] or 1)))
        num_col = max(1, int(float(query.get("num_col", ["2"])[0] or 2)))
        num_row = max(1, int(float(query.get("num_row", ["30"])[0] or 30)))
        shuffle_enabled = query.get("shuffle", ["true"])[0].lower() in ("1", "true", "yes", "on")
        shuffle_seed = query.get("shuffle_seed", ["default"])[0]
        json_first = query.get("json_first", ["true"])[0].lower() in ("1", "true", "yes", "on")
        image_search = query.get("image_search", [""])[0]
        model_prediction_filter = query.get("model_prediction", ["All"])[0] or "All"
        confusion_cell = query.get("confusion_cell", [""])[0].upper()
        confusion_light = query.get("confusion_light", [""])[0]

        if not base_dir:
            self.send_json({"error": "请填写图片根目录。"}, HTTPStatus.BAD_REQUEST)
            return

        base_dir = os.path.abspath(os.path.expanduser(base_dir))
        if not os.path.isdir(base_dir):
            self.send_json({"error": f"图片根目录不存在：{base_dir}"}, HTTPStatus.BAD_REQUEST)
            return

        per_page = num_col * num_row
        annotations = load_annotations()
        annotation_index = build_annotation_index(annotations)
        original_paths = get_filtered_images(light_type, keyword, base_dir)
        original_paths = [
            path for path in original_paths
            if image_matches_search(path, annotations, image_search, annotation_index)
        ]
        if model_prediction_filter != "All":
            original_paths = [
                path for path in original_paths
                if get_model_prediction(path) == model_prediction_filter
            ]
        original_paths = [
            path for path in original_paths
            if matches_confusion_filter(path, annotations, confusion_cell, confusion_light, annotation_index)
        ]
        total_pages = max(1, (len(original_paths) + per_page - 1) // per_page)
        page = min(page, total_pages)
        start = (page - 1) * per_page
        page_paths = original_paths[start:start + per_page]
        if shuffle_enabled:
            page_paths = shuffle_paths(page_paths, shuffle_seed)
        if json_first:
            page_paths = order_json_first(page_paths, annotations, annotation_index)

        items = []
        batch_w, batch_h = 1920, 1080
        for original_path in page_paths:
            display_path = auto_crop_image(original_path, base_dir)
            try:
                with Image.open(display_path) as img:
                    width, height = img.size
                    batch_w, batch_h = width, height
            except Exception:
                width, height = 0, 0

            items.append({
                "id": quote(original_path, safe=""),
                "name": os.path.basename(original_path),
                "modelPrediction": get_model_prediction(original_path),
                "originalPath": original_path,
                "displayPath": display_path,
                "imageUrl": "/api/image?path=" + quote(display_path, safe=""),
                "width": width,
                "height": height,
                "annotation": normalize_annotation(get_annotation_for_image(original_path, annotations, annotation_index)),
            })

        self.send_json({
            "items": items,
            "page": page,
            "totalPages": total_pages,
            "total": len(original_paths),
            "batchWidth": batch_w,
            "batchHeight": batch_h,
            "baseDir": base_dir,
            "shuffle": shuffle_enabled,
            "shuffleSeed": shuffle_seed,
            "jsonFirst": json_first,
            "imageSearch": image_search,
            "modelPredictionFilter": model_prediction_filter,
            "confusionCell": confusion_cell,
            "confusionLight": confusion_light,
        })

    def handle_image(self, query):
        raw_path = query.get("path", [""])[0]
        image_path = os.path.abspath(os.path.expanduser(unquote(raw_path)))
        if not os.path.isfile(image_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        if Path(image_path).suffix.lower() not in IMAGE_EXTENSIONS:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unsupported image type")
            return

        try:
            data = Path(image_path).read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return

        content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_save_annotation(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "评价数据格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        original_path = payload.get("originalPath", "")
        if not original_path:
            self.send_json({"error": "缺少图片路径。"}, HTTPStatus.BAD_REQUEST)
            return

        verdict = CLASSIFICATION_VERDICT_MAP.get(payload.get("verdict", ""), payload.get("verdict", ""))
        if verdict not in ("分类正确", "分类错误"):
            self.send_json({"error": "请先选择“分类正确”或“分类错误”。"}, HTTPStatus.BAD_REQUEST)
            return

        detection_issues = payload.get("detectionIssues", [])
        if not isinstance(detection_issues, list):
            detection_issues = []

        annotation = normalize_annotation({
            "verdict": verdict,
            "greenDefect": payload.get("greenDefect", False),
            "greenDefectRegions": payload.get("greenDefectRegions", []),
            "detectionIssues": detection_issues,
            "missRegions": payload.get("missRegions", []),
            "falseRegions": payload.get("falseRegions", []),
            "note": payload.get("note", ""),
            "imageName": os.path.basename(original_path),
            "updatedAt": payload.get("updatedAt", ""),
        })
        annotations = load_annotations()
        annotations[original_path] = annotation
        save_annotations(annotations)
        self.send_json({"ok": True, "annotation": annotation})

    def handle_annotations_import(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "导入 JSON 格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        imported = annotations_from_import(payload)
        if not imported:
            self.send_json({"error": "没有找到可导入的评价记录。"}, HTTPStatus.BAD_REQUEST)
            return

        save_annotations(imported)
        self.send_json({"ok": True, "imported": len(imported), "total": len(imported)})

    def handle_bulk_default_correct(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "批量保存数据格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        paths = payload.get("paths", [])
        if not isinstance(paths, list) or not paths:
            self.send_json({"error": "缺少当前页图片路径。"}, HTTPStatus.BAD_REQUEST)
            return

        annotations = load_annotations()
        updated_paths = []
        skipped_paths = []

        for original_path in paths:
            if not isinstance(original_path, str) or not original_path:
                continue

            existing = normalize_annotation(get_annotation_for_image(original_path, annotations))
            if existing.get("verdict") in ("分类正确", "分类错误"):
                skipped_paths.append(original_path)
                continue

            annotations[original_path] = normalize_annotation({
                "verdict": "分类正确",
                "greenDefect": False,
                "greenDefectRegions": [],
                "detectionIssues": [],
                "missRegions": [],
                "falseRegions": [],
                "note": payload.get("note", ""),
                "imageName": os.path.basename(original_path),
                "updatedAt": payload.get("updatedAt", ""),
            })
            updated_paths.append(original_path)

        if updated_paths:
            save_annotations(annotations)

        self.send_json({
            "ok": True,
            "updated": len(updated_paths),
            "skipped": len(skipped_paths),
            "updatedPaths": updated_paths,
            "skippedPaths": skipped_paths,
        })

    def handle_annotations_export(self):
        annotations = load_annotations()
        rows = [
            {"originalPath": path, **normalize_annotation(annotation)}
            for path, annotation in annotations.items()
        ]
        self.send_json_download({"count": len(rows), "items": rows}, "inspection_annotations.json")

    def handle_stats(self):
        annotations = load_annotations()
        overall = empty_stats()
        by_light = {}

        for path, annotation in annotations.items():
            accumulate_stats(overall, path, annotation)
            light_type = get_light_type(path)
            by_light.setdefault(light_type, empty_stats())
            accumulate_stats(by_light[light_type], path, annotation)

        self.send_json({
            **finalize_stats(overall),
            "overall": finalize_stats(overall),
            "byLight": {
                light_type: finalize_stats(stats)
                for light_type, stats in sorted(by_light.items())
            },
        })


def main():
    parser = argparse.ArgumentParser(description="Industrial image inspection web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7868, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), InspectionHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Inspection web app running at {url}", flush=True)
    print(f"Annotations: {ANNOTATION_FILE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
