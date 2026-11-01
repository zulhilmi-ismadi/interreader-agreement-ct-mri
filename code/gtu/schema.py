"""Canonical data model: every loader emits one row per (dataset, task, image, rater) in this schema,
and nothing downstream reads raw files. The provenance classes decide floor eligibility. Fields are
part of the released layer's definition, so do not add or rename one without updating every consumer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Provenance(str, Enum):
    """Rater provenance class. Only P1 and P2 may define a floor."""

    P1 = "P1"  # independent human raters, blinded to each other
    P2 = "P2"  # independent non-expert / supervised-crowd raters
    P3 = "P3"  # intra-observer repeat by the same rater
    P4 = "P4"  # sequential unblinded revision
    P5 = "P5"  # algorithm-fused or algorithm-seeded-then-uncorrected
    P6 = "P6"  # single consensus / QC-only reference


#: Only P1 and P2 enter a pooled floor, and they are never merged with each other.
FLOOR_ELIGIBLE = frozenset({Provenance.P1, Provenance.P2})


class Modality(str, Enum):
    MICROSCOPY = "microscopy"
    HISTOLOGY = "histology"
    CYTOLOGY = "cytology"
    FUNDUS = "fundus"
    DERMOSCOPY = "dermoscopy"
    CT = "ct"
    MRI = "mri"
    CXR = "cxr"


#: The NSD tolerance grid splits on this distinction: mm for CT and MRI, px otherwise.
VOLUMETRIC_MODALITIES = frozenset({Modality.CT, Modality.MRI})


class AnnotationType(str, Enum):
    PIXEL = "pixel"
    INSTANCE = "instance"
    BOX = "box"


class Expertise(str, Enum):
    EXPERT = "expert"
    NON_EXPERT = "non_expert"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    RECOMPUTED = "recomputed"  # masks on disk
    PUBLISHED = "published"  # values transcribed from a paper or supplement


UNSTATED_LICENCE = "unstated"


@dataclass(frozen=True)
class Annotation:
    """One row per (dataset, task, image, rater)."""

    dataset: str
    task: str
    image_id: str
    rater_id: str
    mask_path: Path
    spacing: tuple[float, ...] | None
    modality: Modality
    annotation_type: AnnotationType
    rater_provenance: Provenance
    expertise: Expertise
    licence: str
    source_type: SourceType = SourceType.RECOMPUTED
    #: The rater independently corrected an algorithmic pre-segmentation.
    #: Still P1; a sensitivity stratum (seeded against unseeded), never an exclusion.
    seeded: bool = False
    #: Operational field: the higher unit images nest in (patient, video, slide, organ).
    #: Required by the cluster bootstrap, which resamples that unit rather than images.
    cluster_id: str | None = None
    #: Operational field: which value of a multi-class label map this task's binary mask is.
    #: Multi-class maps are split into one binary task per class, and CURVAS ships one file holding
    #: pancreas, kidney and liver together. None means the file is already the task's mask.
    label_value: int | None = None

    def floor_eligible(self) -> bool:
        return self.rater_provenance in FLOOR_ELIGIBLE


def group_by_image(
    annotations: list[Annotation],
) -> dict[tuple[str, str, str], list[Annotation]]:
    """Group into the units a pairwise score is computed over: one image of one task."""
    grouped: dict[tuple[str, str, str], list[Annotation]] = {}
    for ann in annotations:
        grouped.setdefault((ann.dataset, ann.task, ann.image_id), []).append(ann)
    return grouped
