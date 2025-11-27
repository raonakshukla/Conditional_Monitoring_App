# Wind Sense – Conditional Monitoring of Wind Turbine Blades  
### Time, Spectral, Coherence, Transmissibility & Modal Analysis in a Streamlit Dashboard

This repository contains an interactive **Streamlit** application for assessing the structural health of wind turbine blades using operational vibration data.  
The app compares a **reference (Turbine 1)** and a **suspect (Turbine 2)** turbine using:

- Time-domain statistics and moving-window features  
- Welch Power Spectral Density (PSD)  
- Pairwise coherence between sensors  
- Tip-to-root transmissibility in edge/span/flap directions  
- Modal analysis via **Frequency Domain Decomposition (FDD)** and imported **SSI mode shapes**

The interface lets you drill down per blade (B1, B2, B3), location (root/tip), and axis (edge/span/flap) to detect stiffness loss, abnormal damping, changes in load transfer, and blade-specific anomalies.

---

## Repository Structure

```text
WindSense-Wind-Turbine-CMS
├── app.py                          → Streamlit application (main code shown in this repo)
│
├── Turbine_1.csv                   → Vibration dataset for reference / healthy turbine
├── Turbine_2.csv                   → Vibration dataset for suspect / damaged turbine
│   # Columns:
│   #   t, B1_root_edge ... B3_tip_flap (18 acceleration channels + time)
│
├── mode_shapes.xlsx                → Pre-computed modal shapes, natural frequencies & damping
│   # Sheets:
│   #   Mode_Shapes_B1, Mode_Shapes_B2, Mode_Shapes_B3
│
├── requirements.txt                → Python dependencies (streamlit, numpy, pandas, scipy, matplotlib, openpyxl)
└── README.md                       → Project documentation (this file)
