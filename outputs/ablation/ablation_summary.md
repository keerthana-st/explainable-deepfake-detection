# Ablation Study Summary

Target: `dog.jpg`

| Experiment | Global Verdict | Confidence | Regions | Consistency | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Exp 1: Primary System Only** | AI-GENERATED / FAKE | 90.6% | 0 (Susp: 0) | N/A | 1.87s |
| **Exp 2: Primary + Secondary (Classifier-Only Regions)** | AI-GENERATED / FAKE | 90.6% | 2 (Susp: 2) | SUPPORTED | 2.70s |
| **Exp 3: Primary + Secondary + Frequency Regional Fusion** | AI-GENERATED / FAKE | 90.6% | 2 (Susp: 2) | SUPPORTED | 2.32s |
| **Exp 4: Full System (Primary + Secondary + Freq + Grad-CAM)** | AI-GENERATED / FAKE | 90.6% | 2 (Susp: 2) | SUPPORTED | 2.71s |

