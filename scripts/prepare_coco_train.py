"""Prepare COCO train2014 negation data for joint EC+SC training.

Generates:
1. negcap CSV: (image_path, original_caption, negated_caption, negation_type)
   - Object-absence negation using COCO instance annotations
2. MCQ CSV: (correct_answer, caption_0..3, correct_answer_template, image_path)
   - 1 correct caption + 3 negated/cross-image distractors

Usage:
    uv run python scripts/prepare_coco_train.py \
        --coco-dir /path/to/coco \
        --output-dir data/coco_train_negation \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Templates for negated captions
NEGATION_TEMPLATES = {
    "no_X": [
        "{caption}, but there is no {object}",
        "{caption}, and no {object} is present",
        "No {object} is in the scene. {caption}",
    ],
    "without_X": [
        "{caption}, without any {object}",
        "{caption}, without a {object}",
    ],
    "not_X": [
        "A {object} is not present. {caption}",
        "There is no {object} here. {caption}",
    ],
}


def get_article(word: str) -> str:
    """Return 'an' for vowel-starting words, 'a' otherwise."""
    return "an" if word[0].lower() in "aeiou" else "a"


def build_image_categories(instances_data: dict) -> dict[int, set[int]]:
    """Map image_id -> set of category_ids present in the image."""
    img_cats = defaultdict(set)
    for ann in instances_data["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])
    return dict(img_cats)


def build_image_id_to_filename(instances_data: dict) -> dict[int, str]:
    """Map image_id -> filename."""
    return {img["id"]: img["file_name"] for img in instances_data["images"]}


def generate_negcap_row(
    caption: str,
    absent_objects: list[str],
    rng: random.Random,
) -> tuple[str, str]:
    """Generate one negated caption from a list of absent objects.

    Returns (negated_caption, negation_type).
    """
    obj = rng.choice(absent_objects)
    neg_type = rng.choice(["no_X", "without_X", "not_X"])
    templates = NEGATION_TEMPLATES[neg_type]
    template = rng.choice(templates)

    # Clean up caption: remove trailing period for template insertion
    clean_cap = caption.rstrip(". ")
    negated = template.format(caption=clean_cap, object=obj)

    return negated, neg_type


def generate_mcq_row(
    correct_caption: str,
    negated_caption: str,
    cross_captions: list[str],
    rng: random.Random,
) -> tuple[int, list[str], str]:
    """Generate an MCQ row with 1 correct + 3 distractors.

    Returns (correct_answer_idx, [caption_0..3], answer_template).
    """
    distractors = [negated_caption]

    # Add up to 2 cross-image captions as additional distractors
    if len(cross_captions) >= 2:
        distractors.extend(rng.sample(cross_captions, 2))
    elif len(cross_captions) == 1:
        distractors.extend(cross_captions)
        # Duplicate negated with different template if needed
        distractors.append(negated_caption.replace("but there is no", "without any"))
    else:
        distractors.append(negated_caption.replace("but there is no", "without any"))
        distractors.append(negated_caption.replace("but there is no", "and no"))

    captions = distractors[:3] + [correct_caption]
    # Shuffle positions
    indices = list(range(4))
    rng.shuffle(indices)
    shuffled = [captions[i] for i in indices]
    correct_idx = indices.index(3)  # Where did the correct caption end up?

    return correct_idx, shuffled, "positive"


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO negation training data")
    parser.add_argument(
        "--coco-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--output-dir", type=str, default="data/coco_train_negation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negcap-per-caption",
        type=int,
        default=1,
        help="Number of negated captions per original caption",
    )
    parser.add_argument(
        "--max-captions-per-image",
        type=int,
        default=3,
        help="Max original captions to use per image (COCO has 5)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print("Loading COCO annotations...")
    with open(coco_dir / "annotations" / "captions_train2014.json") as f:
        captions_data = json.load(f)
    with open(coco_dir / "annotations" / "instances_train2014.json") as f:
        instances_data = json.load(f)

    # Build lookups
    cat_id_to_name = {c["id"]: c["name"] for c in instances_data["categories"]}
    all_cat_ids = set(cat_id_to_name.keys())
    img_cats = build_image_categories(instances_data)
    img_id_to_file = build_image_id_to_filename(instances_data)

    # Build image_id -> list of captions
    img_captions = defaultdict(list)
    for ann in captions_data["annotations"]:
        img_captions[ann["image_id"]].append(ann["caption"])

    print(f"Images with captions: {len(img_captions)}")
    print(f"Images with objects: {len(img_cats)}")
    print(f"Object categories: {len(cat_id_to_name)}")

    # Build a pool of captions per category for cross-image distractors
    cat_captions = defaultdict(list)
    for img_id, caps in img_captions.items():
        if img_id in img_cats:
            for cat_id in img_cats[img_id]:
                cat_captions[cat_id].extend(caps[:1])  # Just first caption per image

    # Generate negcap and MCQ data
    negcap_rows = []
    mcq_rows = []
    skipped_no_absent = 0

    image_ids = sorted(img_captions.keys())
    rng.shuffle(image_ids)

    for img_id in tqdm(image_ids, desc="Generating"):
        if img_id not in img_cats:
            continue
        if img_id not in img_id_to_file:
            continue

        filename = img_id_to_file[img_id]
        image_path = str(coco_dir / "train2014" / filename)
        present_cats = img_cats[img_id]
        absent_cats = all_cat_ids - present_cats

        if not absent_cats:
            skipped_no_absent += 1
            continue

        absent_objects = [cat_id_to_name[c] for c in absent_cats]
        captions = img_captions[img_id]

        # Limit captions per image
        use_captions = captions[: args.max_captions_per_image]

        # Collect cross-image captions (from images with different categories)
        present_cat_list = list(present_cats)
        cross_caps = []
        for cat_id in present_cat_list[:2]:
            pool = cat_captions.get(cat_id, [])
            if pool:
                cross_caps.extend(rng.sample(pool, min(3, len(pool))))

        for caption in use_captions:
            for _ in range(args.negcap_per_caption):
                negated, neg_type = generate_negcap_row(caption, absent_objects, rng)

                negcap_rows.append(
                    {
                        "image_path": image_path,
                        "original_caption": caption,
                        "negated_caption": negated,
                        "negation_type": neg_type,
                    }
                )

                # Generate corresponding MCQ
                correct_idx, mcq_captions, template = generate_mcq_row(
                    caption, negated, cross_caps, rng
                )
                mcq_rows.append(
                    {
                        "correct_answer": correct_idx,
                        "caption_0": mcq_captions[0],
                        "caption_1": mcq_captions[1],
                        "caption_2": mcq_captions[2],
                        "caption_3": mcq_captions[3],
                        "correct_answer_template": template,
                        "image_path": image_path,
                    }
                )

    print(f"\nGenerated {len(negcap_rows)} negcap pairs")
    print(f"Generated {len(mcq_rows)} MCQ samples")
    print(f"Skipped {skipped_no_absent} images (no absent categories)")

    # Shuffle and save
    rng.shuffle(negcap_rows)
    rng.shuffle(mcq_rows)

    negcap_df = pd.DataFrame(negcap_rows)
    mcq_df = pd.DataFrame(mcq_rows)

    negcap_path = output_dir / "coco_negcap_train2014.csv"
    mcq_path = output_dir / "coco_negmcq_train2014.csv"

    negcap_df.to_csv(negcap_path, index=False)
    mcq_df.to_csv(mcq_path, index=False)

    print(f"\nSaved negcap CSV: {negcap_path} ({len(negcap_df)} rows)")
    print(f"Saved MCQ CSV: {mcq_path} ({len(mcq_df)} rows)")

    # Save metadata
    neg_type_counts = negcap_df["negation_type"].value_counts().to_dict()
    metadata = {
        "num_negcap": len(negcap_df),
        "num_mcq": len(mcq_df),
        "num_images": negcap_df["image_path"].nunique(),
        "negation_type_counts": neg_type_counts,
        "negcap_per_caption": args.negcap_per_caption,
        "max_captions_per_image": args.max_captions_per_image,
        "seed": args.seed,
        "source": "COCO train2014",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nNegation type distribution:")
    for k, v in neg_type_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
