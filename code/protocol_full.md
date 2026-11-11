# V2-188, full-scale verification

Pythia Zenodo 3164691, all 20 light-quark files. Training 0–15 (1.6 M), validation 18–19 (200,000, already examined in the previous phase), new test 16–17 (200,000). Transfer without retraining on the Herwig files 2–3 of Zenodo 3066475 (200,000). The main test is Pythia 16. All files are checked with MD5.

Primary comparison: the KAN of the first study (2,239 parameters) against a standard EFN with a 64×64 MLP encoder and head and a 32-dimensional latent space. The compact EFN of the first study as a secondary control. Same coordinates, pT weights and inputs as the previous networks. Only the actual constituents are processed, without truncation. Five full passes, seed 17, AdamW, batch 1024, lr 0.002 with cosine decay, checkpoint chosen on the mean AUC of the two validation files. No tuning on the tests.

To confirm the net advantage against the wide EFN we require delta AUC of at least 0.003 on the primary, a positive lower bound of the paired 95% bootstrap, and a positive delta also on Pythia 17 and on both Herwig files. FPR at 50% and 80% quark efficiency are descriptive. Per-jet bootstrap conditional on the trained models, without an estimate of the variability between seeds.
