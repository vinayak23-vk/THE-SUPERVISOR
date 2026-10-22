THE SUPERVISOR V0.3 — INSTALL NOTES

Files:
- app.py
- requirements.txt

Before replacing anything:
1. Keep a copy of your current working root app.py.
2. Replace the ROOT app.py with this app.py.
3. Ensure the requirements file used by Streamlit includes yfinance. Your earlier logs showed it used .devcontainer/requirements.txt, so that file may also need yfinance added if Streamlit continues to install from it.
4. Do not use .devcontainer/test_data.py as the app entry point. It is only a test script.
5. Commit changes and wait for deployment.

This app supports Yahoo Finance, CSV upload and synthetic demo data; 1D, 1H, 4H (aggregated from 1H), 15m and 5m.
Yahoo Finance is unofficial and may be delayed, rate-limited or incomplete. No live feed guarantee and no trade execution.
