# Runtime Fixes Plan (Approved)

Status: In Progress

## Issues
1. CSV parsing fails ("Cannot parse file") on some files like data/20_Victims_of_rape.csv
2. Patterns tab: Plotly error "wide-form data with columns of different type" - df_temp has mixed cols/types for histogram

## Plan Steps
1. ✅ Understand code/data - demo1_crime_small.csv good, 20_Victims_of_rape.csv unknown content but CSV.

2. ⏳ Fix safe_load_csv(): Add encodings=['utf-8', 'latin1', 'cp1252'], low_memory=False, engine='python' fallback.

3. ⏳ Fix patterns tab: Always use px.histogram(df, x=pattern_col_num, nbins=24) - derive numeric col first, ensure single x.

4. ⏳ Add try/except around plots with fallback charts.

5. Edit app_fixed_final.py, rerun streamlit, test uploads.

Next: Multiple edit_file calls to app_fixed_final.py.

