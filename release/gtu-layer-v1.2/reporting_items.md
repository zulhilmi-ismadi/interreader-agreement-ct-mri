# Items to report when releasing a multireader segmentation dataset

The eleven reporting items of the article (Table 2), with the example from this study for each.

| Item | What to report | Example from this study |
|---|---|---|
| Readers | Number of readers per image, their training and experience, and a pseudonymous identifier for each reader that stays the same across images and carries no personal data | C3RO: up to 14 experts and 121 non-expert readers, reported as separate groups. PDACVI: four board-certified radiologists with 1, 3, 7 and 22 years' experience, beside the PANORAMA reference label |
| Independence | Whether any reader saw another reader's outlines before or while drawing | LIDC-IDRI: final marks were made after an unblinded review of the other readers' marks |
| Algorithm starting outline | Whether readers began from an algorithm's outline, which algorithm and version, whether the outline was shared, and how much the readers changed it | CURVAS: three radiologists corrected one shared TotalSegmentator outline per scan and agreed with a regenerated version of it nearly as closely as with one another |
| Annotation instructions | The written instructions or criteria the readers followed, and whether they were shared across readers | CURVAS: one radiologist defined the criteria and the other two followed them |
| Image and cohort description | Acquisition parameters, patient characteristics and the split each image belongs to | CURVAS: three scanner groups, with 20 training, 5 validation and 65 test scans |
| Release of each reader's outlines | Each reader's outline for every image, linked to the original image identifier, or a dataset-level summary where a licence does not allow more | Four of the six CT and MRI datasets allow per-image results to be shared |
| Definition of any agreement value | Measure, boundary tolerance, whether pairs or a consensus were compared, how values were averaged over readers and images, and whether images are volumes or single slices | C3RO reports the median over reader pairs; this study reports the mean over images; four QUBIQ 2021 structures are single 2D slices |
| Empty outlines | How pairs are scored when one or both readers drew nothing, and how often this happens | Both-empty pairs are scored as full agreement and counted separately (article, Appendix S3) |
| Uncertainty | Type of interval and what was resampled (images, patients or readers), with the number of each | CURVAS: 95% CIs from resampling the 90 scans, with the three radiologists fixed |
| Evaluation reference | For a challenge or benchmark, whether models are scored against a consensus or an individual reader, on which images, and with what restriction of the scored region | CURVAS: 60 test scans, Dice restricted to consensus regions |
| Licence | Whether derived results may be shared, at the level of each image as well as of the dataset | MSSEG-2016: dataset-level results only, under its data access agreement |

CI = confidence interval.
