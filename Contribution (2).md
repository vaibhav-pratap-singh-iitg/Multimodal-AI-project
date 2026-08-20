# Contribution.md

**Project:** Automatically Discovering and Naming What a Vision Model Doesn't Know
**Dataset:** Waterbirds (via Hugging Face `grodino/waterbirds`)
**Team:** Optimizers (Roles A–D)

This file records what each role built, in the order the pipeline actually
runs. Fill in real names next to each role — left as role labels since that's
how work was tracked throughout.

---

## Role A — Model & Data *(name: Siddhant Patil)*

**Owned:** environment setup, Waterbirds loading, classifier training, `predictions.csv` — the pipeline's single most critical handoff, since every other role's work depends on it.

- Set up the shared repo/Colab and loaded Waterbirds, confirming the four known ground-truth groups (landbird/land, landbird/water, waterbird/land, waterbird/water) before any training began.
- Fine-tuned the vision classifier under audit and produced `predictions.csv` (`image_id, filepath, true_label, pred_label, is_error, split`) for both val and test — confirmed the expected pattern going in: good average accuracy, poor accuracy on mismatched backgrounds.
- Owned split discipline throughout: train/val/test kept strictly separate so later evaluation wasn't contaminated by data used earlier in discovery.
- Ran the anti-circularity check in Week 3: independently verified that CLIP-discovered slice membership actually corresponds to the real background (`place`) metadata, rather than the naming pipeline grading its own homework.

---

## Role B — Embeddings & Clustering *(name:Vaibhav Pratap SIngh)*

**Owned:** CLIP embedding, mean-centering, dimensionality reduction, clustering, and the review grids the whole team judged.

- Selected and swapped CLIP encoders as the project evolved: started on ViT-B/16 (LAION-2B) for the finer patch-size/general-pretraining tradeoff, later upgraded to **ViT-L-14 (laion2b_s32b_b82k)** — confirmed running (768-d embeddings) in the final notebook execution.
- Migrated the data source mid-project from the `wilds` package (filesystem-based) to Hugging Face's `grodino/waterbirds` (in-memory, index-based) — rewrote every image-fetching path in the notebook to match, and flagged the resulting risk (no filename to sanity-check `image_id` alignment against anymore).
- Found and fixed a real bug in the mean-centering step: the original code renormalized embeddings back to unit length after centering, which silently undid the zero-mean property and amplified noise for the most "typical" images. Verified the fix numerically (`max|mean(centered)| ≈ 2.6e-8` in the actual run).
- Added a UMAP dimensionality-reduction stage before clustering (768-d → 40-d in the final run) to counteract the curse of dimensionality in raw CLIP embedding space.
- Iterated on the clustering algorithm itself — tried K-means with the elbow method, settled on **HDBSCAN** for its ability to leave genuine one-off errors as `noise` rather than forcing them into a cluster. Ran a 72-combination hyperparameter sweep (UMAP components/neighbors × `min_cluster_size` × selection method), scored by HDBSCAN's density-based validity index rather than cluster count alone, and selected `min_cluster_size=8` → 14 clusters, 14.7% noise on the 348-image val error set (7.4% error rate on 4715 val images).
- Produced `clusters.csv` (`image_id, cluster, cluster_label`) and the per-cluster image grids for the Week 2 whole-team checkpoint.
- Added cluster visualizations in the final report by adding images of 5-6 of each cluster to help everyone visualize whether the cluster are meaning or not

---

## Role C — Language & Naming *(name:AKshit Arora)*

**Owned:** the word bank, contrastive cluster naming, and pruning the discovered clusters into clean, high-signal slices.

- Built the initial word bank from BLIP captioning + a manual generic-phrase list, ready before any clustering existed (unblocked from day one, per the execution plan).
- Wrote the contrastive naming/scoring code (error-cluster similarity minus correct-image similarity), tested against dummy clusters before real ones existed.
- Named the real 14 discovered clusters, producing `slice_names.csv` (`cluster_id, top_phrase, score, top_5_phrases`).
- Identified and fixed redundancy in the raw output: several of the 14 names were near-duplicates (e.g. two "beach sunset" clusters, three bamboo/tree variants, "boat"/"ship"/"lake" overlap) or weak/low-signal names. Built a dendrogram-based agglomerative merge over CLIP text-embedding cosine distance — chosen specifically because a naive fixed similarity threshold was shown to falsely chain unrelated clusters together (confirmed on this exact cluster set: a naive threshold pulled a weak "head" cluster into the "boat" group through a transitive link).
- Final result: **14 clusters pruned to 8 clean environmental slices** — `beach sunset`, `boat water`, `rock ocean`, `pond`, `lake`, `bamboo forest`, `wooden fence`, `ground woods` — covering 297 of 4715 val images, with 2 clusters (`wooden fence`, `rock ocean`) flagged for manual score-based review rather than auto-dropped.
- Built `final_cluster_assignment.csv`, merging discovered slice names with a ground-truth group fallback (`landbird_on_land` / `landbird_on_water` / `waterbird_on_land` / `waterbird_on_water`) for every val image not swept into a named cluster.

---

## Role D — Evaluation & Fix *(name: Medha Rama Murthy)*

**Owned:** GroupDRO integration, dynamic slice loader socket, evaluation harness, control ablations, and hyperparameter tuning.

- **Dynamic Slice Injection Socket (`cub_dataset.py`)**: Designed and implemented the decoupled slice loader socket in PyTorch dataset code, enabling runtime slice assignment injection via `$env:SLICE_PATH` without hardcoding group labels.
- **Unbiased Evaluation Suite (`eval_on_true_groups.py`)**: Built an independent evaluation harness to measure per-group and worst-group accuracy on **5,794 held-out test images** across the 4 human ground-truth subgroups ($g_{\text{true}} = 2y + \text{place}$).
- **Log Parsing & Metrics Engine (`metrics.py`)**: Developed automated parsing scripts to track average accuracy, worst-group accuracy, and calculate absolute transfer gain ($\Delta$).
- **Control & Ablation Harness**: Implemented the **Random-Slice Control** baseline (10 artificial random group assignments) to empirically evaluate whether natural language slice descriptions are load-bearing.
- **GroupDRO Fine-Tuning & Regularization**: Executed training across all discovered slice iterations (v1, v2, Pruned 0, Pruned 1, and Pruned 2), established optimal $L_2$ regularization parameters ($\lambda = 1.0$) for overparameterized ResNet-50 backbones, and identified early-stopping dynamics (Epochs 40–50) to prevent proxy slice boundary overfitting.



---

