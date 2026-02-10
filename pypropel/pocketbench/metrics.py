"""
Metrics for evaluating binding site predictions.

Implements both geometric localization metrics (DCC, DCA) and
volumetric segmentation metrics (IoU, AP).
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from typing import List, Tuple, Optional, Union
import numpy as np

from .core import PBProtein, PBSite, PBPrediction


# =============================================================================
# Geometric Localization Metrics (Center-based)
# =============================================================================

def compute_dcc(
    prediction: PBPrediction,
    ground_truths: List[PBSite],
    threshold: float = 4.0
) -> Tuple[bool, float]:
    """
    Compute Distance Center-to-Center (DCC) metric.
    
    A prediction is correct if its center is within `threshold` Å
    of ANY valid ground truth site center. This handles the
    UniProt-centric multi-label logic.
    
    Parameters
    ----------
    prediction : PBPrediction
        Predicted binding site.
    ground_truths : List[PBSite]
        List of ground truth sites for the protein.
    threshold : float, optional
        Success cutoff in Angstroms. Default 4.0 Å.
        
    Returns
    -------
    Tuple[bool, float]
        (is_correct, min_distance) - whether prediction is a hit and
        the minimum distance to any ground truth.
        
    Examples
    --------
    >>> pred = PBPrediction(center=[10.0, 10.0, 10.0])
    >>> gts = [PBSite(center=[12.0, 10.0, 10.0])]  # 2Å away
    >>> is_hit, dist = compute_dcc(pred, gts, threshold=4.0)
    >>> print(is_hit, dist)  # True, 2.0
    """
    if not ground_truths:
        return False, float('inf')
    
    pred_center = np.asarray(prediction.center)
    distances = [
        np.linalg.norm(pred_center - np.asarray(gt.center))
        for gt in ground_truths
    ]
    min_dist = min(distances)
    return min_dist <= threshold, min_dist


def compute_dca(
    prediction: PBPrediction,
    ground_truth_atoms: np.ndarray,
    threshold: float = 4.0
) -> Tuple[bool, float]:
    """
    Compute Distance Center-to-Any-Atom (DCA) metric.
    
    A prediction is correct if its center is within `threshold` Å
    of any atom in the ground truth binding site.
    
    Parameters
    ----------
    prediction : PBPrediction
        Predicted binding site.
    ground_truth_atoms : np.ndarray
        Coordinates of all atoms in the ground truth site, shape (M, 3).
    threshold : float, optional
        Success cutoff in Angstroms. Default 4.0 Å.
        
    Returns
    -------
    Tuple[bool, float]
        (is_correct, min_distance) - whether prediction is a hit and
        the minimum distance to any ground truth atom.
    """
    if len(ground_truth_atoms) == 0:
        return False, float('inf')
    
    pred_center = np.asarray(prediction.center).reshape(1, 3)
    gt_atoms = np.asarray(ground_truth_atoms)
    
    # Compute distances to all atoms
    distances = np.linalg.norm(gt_atoms - pred_center, axis=1)
    min_dist = np.min(distances)
    
    return min_dist <= threshold, float(min_dist)


# =============================================================================
# Volumetric Segmentation Metrics (Residue-based)
# =============================================================================

def compute_iou(
    pred_residues: List[int],
    gt_residues: List[int]
) -> float:
    """
    Compute Intersection over Union (IoU) based on residue indices.
    
    This is the Jaccard index between predicted and ground truth
    residue sets.
    
    Parameters
    ----------
    pred_residues : List[int]
        0-indexed residue indices from prediction.
    gt_residues : List[int]
        0-indexed residue indices from ground truth.
        
    Returns
    -------
    float
        IoU score in range [0, 1].
        
    Examples
    --------
    >>> compute_iou([0, 1, 2, 3], [2, 3, 4, 5])
    0.333...  # 2 overlap, 6 union
    """
    pred_set = set(pred_residues)
    gt_set = set(gt_residues)
    
    if not pred_set and not gt_set:
        return 1.0  # Both empty = perfect match
    if not pred_set or not gt_set:
        return 0.0  # One empty = no match
    
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    
    return intersection / union


def compute_ap(
    predictions: List[PBPrediction],
    ground_truths: List[PBSite],
    iou_threshold: float = 0.5,
    protein_coords: Optional[np.ndarray] = None,
    expand_radius: float = 9.0
) -> float:
    """
    Compute Average Precision at a specific IoU threshold (AP@IoU).
    
    For predictions without residue lists (center-only models like VN-EGNN),
    automatically expands centers to residue lists using radius fallback.
    
    Parameters
    ----------
    predictions : List[PBPrediction]
        List of predictions sorted by confidence (descending).
    ground_truths : List[PBSite]
        List of ground truth binding sites.
    iou_threshold : float, optional
        IoU threshold for considering a match. Default 0.5 (AP@50).
    protein_coords : np.ndarray, optional
        C-alpha coordinates for radius expansion fallback, shape (N, 3).
    expand_radius : float, optional
        Radius for expanding center-only predictions. Default 9.0 Å.
        
    Returns
    -------
    float
        Average Precision score in range [0, 1].
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # Sort predictions by confidence (should already be sorted)
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)
    
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    
    total_gt = len(ground_truths)
    
    for pred in sorted_preds:
        # Expand prediction if no residues
        pred_residues = pred.residues
        if not pred_residues and protein_coords is not None:
            pred_residues = expand_center_to_residues(
                pred.center, protein_coords, expand_radius
            )
        
        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou(pred_residues, gt.residues)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match meets threshold
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
        
        precision = tp / (tp + fp)
        recall = tp / total_gt
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute AP using 11-point interpolation or all-point
    if not precisions:
        return 0.0
    
    # All-point AP (area under PR curve)
    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precisions, recalls):
        ap += prec * (rec - prev_recall)
        prev_recall = rec
    
    return ap


# =============================================================================
# Utility Functions
# =============================================================================

def expand_center_to_residues(
    center: np.ndarray,
    coords: np.ndarray,
    radius: float = 9.0
) -> List[int]:
    """
    Expand a center point to residue list using radius cutoff.
    
    This is the "9Å radius fallback" for models (like VN-EGNN) that
    only output center coordinates without explicit residue predictions.
    
    Parameters
    ----------
    center : np.ndarray
        Predicted center [x, y, z].
    coords : np.ndarray
        C-alpha coordinates of all residues, shape (N, 3).
    radius : float, optional
        Radius cutoff in Angstroms. Default 9.0 Å.
        
    Returns
    -------
    List[int]
        0-indexed residue indices within radius of center.
        
    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [5, 0, 0], [15, 0, 0]])
    >>> expand_center_to_residues([0, 0, 0], coords, radius=10.0)
    [0, 1]  # Residues at 0Å and 5Å, but not 15Å
    """
    center = np.asarray(center).reshape(1, 3)
    coords = np.asarray(coords)
    
    distances = np.linalg.norm(coords - center, axis=1)
    residue_indices = np.where(distances <= radius)[0].tolist()
    
    return residue_indices


def compute_site_center(
    coords: np.ndarray,
    residue_indices: List[int]
) -> np.ndarray:
    """
    Compute geometric center of a binding site from residue coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        C-alpha coordinates of all residues, shape (N, 3).
    residue_indices : List[int]
        0-indexed residue indices belonging to the site.
        
    Returns
    -------
    np.ndarray
        Geometric center [x, y, z].
    """
    if not residue_indices:
        return np.zeros(3, dtype=np.float32)
    
    site_coords = coords[residue_indices]
    return np.mean(site_coords, axis=0)


# =============================================================================
# Prediction Filtering Utilities
# =============================================================================

# Common crystallographic artifact ligand codes that are NOT biologically
# relevant binding sites. These appear as HETATM in PDB files but are
# buffer components, cryoprotectants, or crystallization additives.
CRYSTALLOGRAPHIC_ARTIFACTS = {
    # Buffers & solvents
    'GOL', 'EDO', 'PEG', 'DMS', 'ACT', 'BME', 'TRS', 'MES', 'EPE',
    'MPD', 'IPA', 'PGE', 'PG4', 'P6G', '1PE', '2PE',
    # Ions & small inorganics  
    'SO4', 'PO4', 'CL', 'BR', 'IOD', 'NO3', 'SCN', 'ACY', 'FMT',
    'MLI', 'NH4', 'CIT', 'TAR', 'OXL',
    # Detergents
    'BOG', 'LDA', 'SDS', 'LMT', 'OLC',
    # Common cryoprotectants
    'XYL', 'SUC', 'GLC', 'MAL', 'TRE',
}


def filter_predictions_by_gt(
    predictions: List[PBPrediction],
    ground_truths: List[PBSite]
) -> List[PBPrediction]:
    """
    Keep only predictions whose ligand_id matches a ground truth site.
    
    This implements "Approach 1: GT-Matched AP" — testing whether the
    model can correctly predict the pocket when given the right ligand.
    
    Parameters
    ----------
    predictions : List[PBPrediction]
        All predictions (may include non-GT ligands).
    ground_truths : List[PBSite]
        Ground truth sites with ligand_id set.
        
    Returns
    -------
    List[PBPrediction]
        Filtered predictions matching GT ligand codes.
    """
    gt_ligands = {gt.ligand_id for gt in ground_truths if gt.ligand_id}
    if not gt_ligands:
        return predictions  # No GT ligand info, return all
    
    return [p for p in predictions if p.ligand_id in gt_ligands]


def filter_artifact_predictions(
    predictions: List[PBPrediction],
    artifact_codes: Optional[set] = None
) -> List[PBPrediction]:
    """
    Remove predictions from known crystallographic artifact ligands.
    
    Parameters
    ----------
    predictions : List[PBPrediction]
        All predictions.
    artifact_codes : set, optional
        Custom artifact codes. Defaults to CRYSTALLOGRAPHIC_ARTIFACTS.
        
    Returns
    -------
    List[PBPrediction]
        Predictions with artifacts removed.
    """
    if artifact_codes is None:
        artifact_codes = CRYSTALLOGRAPHIC_ARTIFACTS
    
    return [p for p in predictions if p.ligand_id not in artifact_codes]


def deduplicate_predictions(
    predictions: List[PBPrediction],
    iou_threshold: float = 0.5
) -> List[PBPrediction]:
    """
    Merge overlapping predictions, keeping higher-confidence ones.
    
    When multiple predictions target the same pocket (IoU >= threshold),
    only the highest-confidence prediction is kept.
    
    Parameters
    ----------
    predictions : List[PBPrediction]
        Predictions sorted by confidence (descending).
    iou_threshold : float
        IoU threshold for considering two predictions as duplicates.
        
    Returns
    -------
    List[PBPrediction]
        Deduplicated predictions.
    """
    if not predictions:
        return []
    
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    kept = []
    
    for pred in sorted_preds:
        is_duplicate = False
        for existing in kept:
            if not pred.residues or not existing.residues:
                # Fall back to center distance if no residues
                dist = np.linalg.norm(
                    np.asarray(pred.center) - np.asarray(existing.center)
                )
                if dist < 4.0:  # Same pocket if centers within 4Å
                    is_duplicate = True
                    break
            else:
                iou = compute_iou(pred.residues, existing.residues)
                if iou >= iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            kept.append(pred)
    
    return kept


# =============================================================================
# Batch Evaluation Functions
# =============================================================================

def evaluate_predictions(
    proteins: List[PBProtein],
    predictions: List[List[PBPrediction]],
    dcc_threshold: float = 4.0,
    iou_threshold: float = 0.5
) -> dict:
    """
    Evaluate predictions across multiple proteins.
    
    Computes three AP variants:
    - mean_ap: All predictions (backward-compatible, may be inflated by artifacts)
    - mean_ap_matched: Only predictions matching GT ligand codes
    - mean_ap_filtered: Artifact ligands removed + overlapping predictions deduplicated
    
    Parameters
    ----------
    proteins : List[PBProtein]
        List of proteins with ground truth sites.
    predictions : List[List[PBPrediction]]
        Predictions for each protein.
    dcc_threshold : float, optional
        Distance threshold for DCC metric. Default 4.0 Å.
    iou_threshold : float, optional
        IoU threshold for AP calculation. Default 0.5.
        
    Returns
    -------
    dict
        Evaluation results with keys:
        - 'dcc_success_rate': fraction of proteins with at least one DCC hit
        - 'dcc_top1_rate': fraction where top-1 prediction is a DCC hit
        - 'mean_iou': mean IoU across all prediction-ground truth pairs
        - 'mean_ap': mean AP across proteins (all predictions)
        - 'mean_ap_matched': mean AP using only GT-matching ligand predictions
        - 'mean_ap_filtered': mean AP with artifacts removed + dedup
    """
    dcc_hits = 0
    dcc_top1_hits = 0
    all_ious = []
    all_aps = []
    all_aps_matched = []
    all_aps_filtered = []
    
    for protein, preds in zip(proteins, predictions):
        if not preds:
            continue
        
        gts = protein.ground_truth_sites
        if not gts:
            continue
        
        # DCC for top-1 prediction
        is_hit, _ = compute_dcc(preds[0], gts, dcc_threshold)
        if is_hit:
            dcc_top1_hits += 1
        
        # DCC for any prediction
        any_hit = any(
            compute_dcc(pred, gts, dcc_threshold)[0]
            for pred in preds
        )
        if any_hit:
            dcc_hits += 1
        
        # IoU for top-1 vs best GT
        if preds[0].residues and gts[0].residues:
            best_iou = max(
                compute_iou(preds[0].residues, gt.residues)
                for gt in gts
            )
            all_ious.append(best_iou)
        
        # AP (all predictions — backward-compatible)
        ap = compute_ap(
            preds, gts, 
            iou_threshold=iou_threshold,
            protein_coords=protein.coords
        )
        all_aps.append(ap)
        
        # AP matched (only GT-matching ligands)
        matched_preds = filter_predictions_by_gt(preds, gts)
        if matched_preds:
            ap_matched = compute_ap(
                matched_preds, gts,
                iou_threshold=iou_threshold,
                protein_coords=protein.coords
            )
            all_aps_matched.append(ap_matched)
        else:
            all_aps_matched.append(0.0)
        
        # AP filtered (artifacts removed + deduplicated)
        filtered_preds = filter_artifact_predictions(preds)
        filtered_preds = deduplicate_predictions(filtered_preds, iou_threshold=iou_threshold)
        if filtered_preds:
            ap_filtered = compute_ap(
                filtered_preds, gts,
                iou_threshold=iou_threshold,
                protein_coords=protein.coords
            )
            all_aps_filtered.append(ap_filtered)
        else:
            all_aps_filtered.append(0.0)
    
    n_proteins = len(proteins)
    
    return {
        'dcc_success_rate': dcc_hits / n_proteins if n_proteins > 0 else 0.0,
        'dcc_top1_rate': dcc_top1_hits / n_proteins if n_proteins > 0 else 0.0,
        'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        'mean_ap': np.mean(all_aps) if all_aps else 0.0,
        'mean_ap_matched': np.mean(all_aps_matched) if all_aps_matched else 0.0,
        'mean_ap_filtered': np.mean(all_aps_filtered) if all_aps_filtered else 0.0,
        'n_proteins': n_proteins,
    }
