# Final revision evidence

- Visual GRU: 0.3034 +/- 0.0113 event mAP within +/-10 s.
- Late fusion: 0.3144 +/- 0.0029; mean paired gain +0.0110.
- Symmetric future-text control: 0.3507 +/- 0.0046; this quantifies optimistic leakage when future commentary is admitted.
- Zero-text retraining: 0.3008; 60-s text shift: 0.2959.
- TF-IDF late fusion: 0.3124.
- The paired game-cluster bootstrap supports the modest late-fusion gain, while the five-seed t interval remains wider and should be reported as training-instability uncertainty.
- Residual graph message passing and latent-flow regularization do not improve upon simple late fusion.
