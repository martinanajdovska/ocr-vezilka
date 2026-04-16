- phase1_output/summary.json and phase1_output/total_statistics.json contain the alignment data and statistics for all books combined
- summary for train, test and val splits can be found in the corresponding folders in phase1_output
- summary for each book individually can be found in their corresponding folders inside train/test/val folders
-phase 2 performs a detailed alignment-based analysis of OCR errors in Macedonian Cyrillic books. The combined alignment data and statistics are stored in phase1_output/summary.json and phase1_output/total_statistics.json, while split-level and book-level summaries are available inside the corresponding train, test, and val folders.
-phase 3 builds synthetic noise generators based on the observed OCR error patterns and evaluates how closely they match real OCR noise. The goal is to approximate substitutions, insertions, deletions, and structural distortions such as word splits, merges, diacritic loss, and punctuation changes as realistically as possible.

