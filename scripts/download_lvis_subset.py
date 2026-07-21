"""Download a small LVIS v1 validation subset and its COCO source images."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
import tempfile
from urllib.parse import urlparse

from PIL import Image

ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "text-video-annotator/0.1"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def download_image(url: str, destination: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            temporary.write_bytes(download_bytes(url))
            if not valid_image(temporary):
                raise OSError("downloaded bytes are not a readable image")
            temporary.replace(destination)
            return
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lvis/raw"))
    parser.add_argument("--max-images", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    existing_annotation = args.output / "lvis_v1_val_subset.json"
    if existing_annotation.is_file():
        print(f"Using existing subset annotations: {existing_annotation}")
        payload = json.loads(existing_annotation.read_text(encoding="utf-8"))
        if len(payload.get("images", [])) < args.max_images:
            payload = None
    else:
        payload = None
    if payload is None:
        with tempfile.TemporaryDirectory(prefix="lvis_download_") as temp:
            archive = Path(temp) / "lvis_val.zip"
            print(f"Downloading LVIS annotations from {ANNOTATIONS_URL}")
            archive.write_bytes(download_bytes(ANNOTATIONS_URL))
            with ZipFile(archive) as zipped:
                json_name = next(name for name in zipped.namelist() if name.endswith(".json"))
                payload = json.loads(zipped.read(json_name))
    categories = {item["id"]: item["name"] for item in payload.get("categories", [])}
    image_ids = {annotation["image_id"] for annotation in payload.get("annotations", [])}
    selected = [image for image in payload["images"] if image["id"] in image_ids][: args.max_images]
    selected_ids = {image["id"] for image in selected}
    selected_annotations = [annotation for annotation in payload["annotations"] if annotation["image_id"] in selected_ids]
    selected_payload = {**payload, "images": selected, "annotations": selected_annotations}
    annotation_path = args.output / "lvis_v1_val_subset.json"
    annotation_path.write_text(json.dumps(selected_payload), encoding="utf-8")
    for index, image in enumerate(selected, 1):
        file_name = image.get("file_name") or Path(urlparse(image["coco_url"]).path).name
        destination = image_dir / file_name
        if valid_image(destination):
            continue
        print(f"[{index}/{len(selected)}] {file_name}")
        # Preserve the official COCO URL. In particular, do not rewrite HTTP
        # to HTTPS: some Windows certificate stores reject the redirected host.
        download_image(image["coco_url"], destination)
    manifest = {"dataset": "lvis", "annotation": str(annotation_path), "images": [
        {"id": image["id"], "file_name": image.get("file_name") or Path(urlparse(image["coco_url"]).path).name, "path": str(image_dir / (image.get("file_name") or Path(urlparse(image["coco_url"]).path).name)),
         "categories": sorted({categories[a["category_id"]] for a in selected_annotations if a["image_id"] == image["id"]})}
        for image in selected
    ]}
    failed = [item["path"] for item in manifest["images"] if not valid_image(Path(item["path"]))]
    (args.output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failed:
        raise RuntimeError(f"Missing or unreadable downloaded images: {failed}")
    print(f"Downloaded {len(selected)} images to {image_dir}")


if __name__ == "__main__":
    main()
