# 2026 EY AI & Data Challenge — Water Quality Prediction

## What Is This Project?

We're participating in the 2026 EY AI & Data Challenge, a global competition focused on predicting water quality in South African rivers. Nearly 2.2 billion people worldwide lack access to clean water. This challenge asks us to build AI tools that can help monitor and predict water quality to protect communities.

## What Are We Predicting?

Three measurements that tell us how healthy river water is:

1. **Total Alkalinity** — The water's ability to resist becoming acidic. Think of it as the water's immune system against acid. Healthy range: 20-200 mg/L.

2. **Electrical Conductance** — How many dissolved salts and minerals are in the water. Too many dissolved substances can make water unsafe or undrinkable. Healthy range: below 800 uS/cm.

3. **Dissolved Reactive Phosphorus** — How much phosphorus is dissolved in the water. Too much phosphorus causes algae to grow out of control, which kills fish and makes water toxic. Healthy range: below 100 ug/L.

## The Challenge

We're given water quality measurements from 162 river sampling locations across South Africa, collected over 5 years (2011-2015). Our job is to predict water quality at 24 completely new locations we've never seen before.

We can't just memorize what each location looks like. We need to understand **why** water quality varies from place to place. To do that, we combine multiple data sources to build a complete environmental picture of each river location.

## Our Approach

We combine several publicly available data sources to understand the environment around each river sampling point:

- **Satellite imagery** (Landsat) — to see vegetation density, urban areas, and farmland near each river
- **Climate data** (TerraClimate) — rainfall, temperature, soil moisture, drought conditions, and more
- **Terrain data** (SRTM) — elevation, slope, and how water flows across the landscape
- **Land use maps** (ESA WorldCover) — what percentage of land around the river is crops, forest, or city
- **Soil data** (SoilGrids) — soil type, acidity, and chemical composition

Our model learns the relationship between these environmental factors and water quality, then applies that knowledge to predict quality at new, unseen locations.

## How Is Our Score Calculated?

Our score is the R-squared value, a statistical measure of how well predictions match reality:

- **1.0** means perfect predictions
- **0.0** means the model is no better than just guessing the average
- Below 0 means the model is worse than guessing

The final score is the average R-squared across all three water quality measurements. The starting benchmark model scores 0.20. Our target is 0.91 or higher.

## Project Structure

```
EY-Water-ML-Challenge/
├── Resources/              # Original files from EY (do not edit)
├── datasets/               # Working copies of data files
│   ├── processed/          # Cleaned and enriched datasets
│   └── external/           # Additional data sources we fetch
├── notebooks/              # Jupyter notebooks for analysis
│   └── reference/          # Copies of EY-provided notebooks
├── src/                    # Python source code
├── configs/                # Model configuration files
└── outputs/                # Submissions, saved models, figures
```

## Key Dates

- **March 13, 2026** — Submission deadline
- **April 1, 2026** — Finalists announced
- **May 6, 2026** — Awards ceremony

## Data Sources

- Water quality data: [UNEP GEMStat](https://gemstat.org/)
- Landsat satellite imagery: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2)
- TerraClimate: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/terraclimate)
- Elevation data: [SRTM via OpenTopography](https://www.opentopography.org/)
- Land cover: [ESA WorldCover](https://esa-worldcover.org/)
- Soil data: [SoilGrids](https://soilgrids.org/)
