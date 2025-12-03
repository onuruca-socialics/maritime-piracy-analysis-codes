# Install required packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Generate plots

## Cramer's V test

```bash
python stat_analysis chi data-2015-2024-v2.csv
```

## Region based analysis

```bash
python stat_analysis rel data-2015-2024-v2.csv
```

## GIS data analysis

```bash
python stat_analysis gis gis_data.csv
```
