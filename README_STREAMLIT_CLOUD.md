# Streamlit Cloud Deployment Guide

This guide helps you deploy the PFAS Removal DFT GUI to Streamlit Cloud.

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free tier available)
3. Repository with your code

## Deployment Steps

1. **Push your code to GitHub**
   - Make sure all files are committed
   - Push to your GitHub repository

2. **Connect to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configuration**
   - The app uses `requirements.txt` for dependencies
   - Streamlit Cloud will automatically install packages
   - No external software downloads needed (everything comes from PyPI)

## File Structure for Streamlit Cloud

```
.
├── streamlit_app.py          # Main app (required)
├── requirements.txt          # Python dependencies (required)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── src/
│   └── pfasdft/             # Application modules
├── artifacts/                # Reference data and configs
└── README.md                 # Documentation
```

## Important Notes

- **No external executables**: The app only uses Python packages from PyPI
- **File uploads**: All file operations use temporary files that are cleaned up
- **RDKit**: Installed via pip (may take a few minutes on first deploy)
- **Artifacts**: Reference data is included in the repository

## Troubleshooting

If you encounter errors:

1. Check Streamlit Cloud logs (Manage app → Logs)
2. Verify all imports are available in requirements.txt
3. Ensure file paths are relative (not absolute)
4. Check that temporary file operations are properly handled

## Dependencies

All dependencies are listed in `requirements.txt`:
- rdkit (for molecular structure handling)
- pandas (for data processing)
- numpy (for numerical operations)
- streamlit (web framework)

No external software (like ORCA) needs to be installed - the app only generates input files and parses outputs.
