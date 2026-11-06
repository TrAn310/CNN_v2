
from pathlib import Path
from itertools import combinations
import csv

import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "evaluation_results"
CHART_DIR = RESULT_DIR / "verification_charts"

EMBEDDING_FILE = RESULT_DIR / "verification_embeddings.npz"

THRESHOLD = 0.32
THRESHOLD_STEP = 0.01

CHART_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. METRIC FUNCTIONS
# ==================================================

def calculate_metrics(y_true, y_pred):

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "FAR": far,
        "FRR": frr,
        "Accuracy": accuracy
    }


# ==================================================
# 3. LOAD EMBEDDINGS
# ==================================================

def load_data():

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    data = np.load(EMBEDDING_FILE)

    embeddings = data["embeddings"].astype(np.float32)
    participants = data["participants"]
    image_names = data["image_names"]

    if len(embeddings) != len(participants):
        raise ValueError("Embedding and participant counts do not match.")

    if len(embeddings) != len(image_names):
        raise ValueError("Embedding and image name counts do not match.")

    if len(embeddings) < 2:
        raise ValueError("At least two images are required.")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain invalid numeric values.")

    print("\n[INFO] Data loaded successfully.")
    print(f"[INFO] Number of images: {len(embeddings)}")
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    print(f"[INFO] Participants: {np.unique(participants).tolist()}")

    return embeddings, participants, image_names


# ==================================================
# 4. GENERATE ALL IMAGE PAIRS
# ==================================================

def generate_pairs(embeddings, participants, image_names):

    pair_records = []

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError("One or more embeddings have zero norm.")

    normalized = embeddings / norms[:, np.newaxis]

    for i, j in combinations(range(len(embeddings)), 2):

        similarity = float(
            np.dot(normalized[i], normalized[j])
        )

        same_person = int(
            participants[i] == participants[j]
        )

        pair_records.append({
            "image1": str(image_names[i]),
            "participant1": str(participants[i]),
            "image2": str(image_names[j]),
            "participant2": str(participants[j]),
            "similarity": similarity,
            "actual": same_person
        })

    return pair_records


# ==================================================
# 5. SAVE PAIRWISE CSV
# ==================================================

def save_pairwise_csv(records):

    output_file = RESULT_DIR / "verification_pairwise_results.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image1",
                "participant1",
                "image2",
                "participant2",
                "similarity",
                "actual",
                "predicted"
            ]
        )

        writer.writeheader()

        for record in records:

            predicted = int(
                record["similarity"] >= THRESHOLD
            )

            writer.writerow({
                "image1": record["image1"],
                "participant1": record["participant1"],
                "image2": record["image2"],
                "participant2": record["participant2"],
                "similarity": f'{record["similarity"]:.6f}',
                "actual": record["actual"],
                "predicted": predicted
            })

    return output_file


# ==================================================
# 6. THRESHOLD EVALUATION
# ==================================================

def evaluate_thresholds(records):

    thresholds = np.round(
        np.arange(0, 1.0001, THRESHOLD_STEP),
        2
    )

    y_true = np.array(
        [r["actual"] for r in records]
    )

    similarities = np.array(
        [r["similarity"] for r in records]
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            similarities >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        results.append({
            "threshold": float(threshold),
            **metrics
        })

    output_file = RESULT_DIR / "verification_threshold_results.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "TP",
                "TN",
                "FP",
                "FN",
                "FAR",
                "FRR",
                "Accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    return results, output_file


# ==================================================
# 7. CHART 1: SIMILARITY DISTRIBUTION
# ==================================================

def plot_similarity_distribution(records):

    same_scores = [
        r["similarity"]
        for r in records
        if r["actual"] == 1
    ]

    different_scores = [
        r["similarity"]
        for r in records
        if r["actual"] == 0
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        same_scores,
        bins=12,
        alpha=0.6,
        label="Same person"
    )

    plt.hist(
        different_scores,
        bins=12,
        alpha=0.6,
        label="Different person"
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label=f"Threshold = {THRESHOLD}"
    )

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of pairs")
    plt.title("Face Verification - Similarity Distribution")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = CHART_DIR / "01_similarity_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    return output


# ==================================================
# 8. CHART 2: FAR AND FRR
# ==================================================

def plot_far_frr(results):

    thresholds = [r["threshold"] for r in results]
    far = [r["FAR"] * 100 for r in results]
    frr = [r["FRR"] * 100 for r in results]

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        far,
        label="FAR"
    )

    plt.plot(
        thresholds,
        frr,
        label="FRR"
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label=f"Selected threshold = {THRESHOLD}"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Error Rate (%)")
    plt.title("FAR and FRR vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output = CHART_DIR / "02_far_frr_curve.png"
    plt.savefig(output, dpi=300)
    plt.close()

    return output


# ==================================================
# 9. CHART 3: CONFUSION MATRIX
# ==================================================

def plot_confusion_matrix(metrics):

    matrix = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]]
    ])

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(matrix)

    labels = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{matrix[i, j]}",
                ha="center",
                va="center",
                fontsize=14
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Different person",
        "Same person"
    ])

    ax.set_yticklabels([
        "Different person",
        "Same person"
    ])

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_title(
        f"Confusion Matrix (Threshold = {THRESHOLD})"
    )

    plt.tight_layout()

    output = CHART_DIR / "03_confusion_matrix.png"
    plt.savefig(output, dpi=300)
    plt.close()

    return output


# ==================================================
# 10. CHART 4: SUMMARY DASHBOARD
# ==================================================

def plot_summary(metrics, records, image_count):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    total = len(records)

    same_count = sum(
        r["actual"] == 1 for r in records
    )

    different_count = total - same_count

    summary = [
        ["EVALUATION ITEM", "RESULT"],
        ["Number of images", str(image_count)],
        ["Total image pairs", str(total)],
        ["Same-person pairs", str(same_count)],
        ["Different-person pairs", str(different_count)],
        ["Threshold", f"{THRESHOLD:.2f}"],
        ["True Positive (TP)", str(metrics["TP"])],
        ["True Negative (TN)", str(metrics["TN"])],
        ["False Positive (FP)", str(metrics["FP"])],
        ["False Negative (FN)", str(metrics["FN"])],
        ["Accuracy", f'{metrics["Accuracy"] * 100:.2f}%'],
        ["FAR", f'{metrics["FAR"] * 100:.2f}%'],
        ["FRR", f'{metrics["FRR"] * 100:.2f}%']
    ]

    table = ax.table(
        cellText=summary,
        cellLoc="center",
        loc="center",
        colWidths=[0.55, 0.35]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    ax.set_title(
        "FACE VERIFICATION - EVALUATION SUMMARY",
        fontsize=15,
        pad=20
    )

    plt.tight_layout()

    output = CHART_DIR / "04_evaluation_summary.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    return output


# ==================================================
# 11. PRINT TERMINAL REPORT
# ==================================================

def print_report(
    image_count,
    records,
    metrics,
    threshold_file,
    pairwise_file,
    chart_files
):

    same_count = sum(
        r["actual"] == 1 for r in records
    )

    different_count = len(records) - same_count

    print("\n")
    print("=" * 60)
    print("FACE VERIFICATION - EVALUATION REPORT")
    print("=" * 60)

    print("\n[DATASET]")
    print(f"Number of images:          {image_count}")
    print(f"Total image pairs:         {len(records)}")
    print(f"Same-person pairs:         {same_count}")
    print(f"Different-person pairs:    {different_count}")

    print("\n[CONFUSION MATRIX]")
    print(f"True Positive  (TP):       {metrics['TP']}")
    print(f"True Negative  (TN):       {metrics['TN']}")
    print(f"False Positive (FP):       {metrics['FP']}")
    print(f"False Negative (FN):       {metrics['FN']}")

    print("\n[PERFORMANCE METRICS]")
    print(f"Threshold:                 {THRESHOLD:.2f}")
    print(f"Accuracy:                  {metrics['Accuracy'] * 100:.2f}%")
    print(f"FAR:                       {metrics['FAR'] * 100:.2f}%")
    print(f"FRR:                       {metrics['FRR'] * 100:.2f}%")

    print("\n[OUTPUT FILES]")
    print(f"Pairwise CSV:              {pairwise_file}")
    print(f"Threshold CSV:             {threshold_file}")

    for chart in chart_files:
        print(f"Chart:                     {chart}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


# ==================================================
# 12. MAIN
# ==================================================

def main():

    print("=" * 60)
    print("STARTING FACE VERIFICATION EVALUATION")
    print("=" * 60)

    embeddings, participants, image_names = load_data()

    records = generate_pairs(
        embeddings,
        participants,
        image_names
    )

    print(f"\n[INFO] Generated {len(records)} image pairs.")

    pairwise_file = save_pairwise_csv(records)

    y_true = np.array(
        [r["actual"] for r in records]
    )

    y_pred = np.array([
        int(r["similarity"] >= THRESHOLD)
        for r in records
    ])

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    threshold_results, threshold_file = evaluate_thresholds(
        records
    )

    chart_files = []

    chart_files.append(
        plot_similarity_distribution(records)
    )

    chart_files.append(
        plot_far_frr(threshold_results)
    )

    chart_files.append(
        plot_confusion_matrix(metrics)
    )

    chart_files.append(
        plot_summary(
            metrics,
            records,
            len(embeddings)
        )
    )

    print_report(
        len(embeddings),
        records,
        metrics,
        threshold_file,
        pairwise_file,
        chart_files
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print(f"\n[ERROR] Evaluation failed: {error}")
