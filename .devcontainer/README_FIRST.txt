THE SUPERVISOR V0.2 — FIXED BUILD

This build fixes the broken checks list and incorrect EMA variable:
- closes the Support bounce checks list before `elif`
- uses `above_ema100` consistently
- keeps all setup branches syntactically valid
- uses supported Streamlit dataframe/chart width arguments

On GitHub, replace the ENTIRE contents of the existing app.py with this app.py.
Do not paste this README or requirements into app.py.
Keep requirements.txt as a separate file.
Commit changes, then wait for Streamlit Cloud to redeploy and refresh the app.

This is a technical-analysis checklist only. It does not fetch live data, connect to a broker, place orders, or guarantee outcomes.
