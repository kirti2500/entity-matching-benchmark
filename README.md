# Consultbae AI Automation Assignment

**Author:** [Your Name]

## Structure
```
Consultbae-ai-automation-assignment/
├── data/               # raw + cleaned CSVs
├── scripts/            # Task 1: merge pipeline
├── automation/          # Task 2: n8n flow export
├── audio-app/            # Task 3: browser audio app
├── REPORT.md            # Task 4: data issues report
└── README.md            # this file
```

## How to run each task

### Task 1 — Merge
```
cd scripts
python merge_pipeline.py
```
Reads the 3 source CSVs, normalizes phone/email/city/CTC/dates, and merges
duplicate people across files using a union-find match on normalized phone
OR email. Outputs:
- `data/master_database.csv` — 61 unique people (from 103 raw rows)
- `data/all_rows_normalized.csv` — full audit trail of every row before merge

### Task 2 — No-code automation (n8n)
Run n8n locally:
```
npx n8n
```
Open `http://localhost:5678`, then **Import from File** → select
`automation/flow.json`. The flow implements a duplicate-alert check on new
CSV rows (matches on phone number). A screenshot of the flow is also
included in `automation/` in case n8n isn't run by the grader.

### Task 3 — Audio intake app
Open `audio-app/audio_app.html` directly in a browser (Chrome recommended,
for mic access). No server or install needed.
- Enter name + phone
- Upload an audio file, or click "Start Recording" to record via mic
- Click "Analyze Audio" — extracts duration, sample rate, estimated bitrate,
  and loudness (RMS, in dB)

### Task 4 — Data issues report
See `REPORT.md` for the full write-up of every data quality issue found and
how it was handled.

## Key assumptions / judgment calls
- **CTC unit ambiguity:** values under 100 in the "Current CTC" column were
  assumed to be LPA (lakhs) and converted to rupees; values ≥100 were assumed
  already in rupees.
- **Rate conversion:** hourly rates in `source2` were converted to a monthly
  equivalent assuming 8 hrs/day × 22 working days/month.
- **Ambiguous duplicate:** two "Arjun Mehta" entries in `source3` have
  different phone numbers and were kept as separate people rather than
  merged, since there wasn't enough signal to be certain they're the same
  person. See REPORT.md for full reasoning.
- **Cross-file matching:** `source2` and `source3` don't share phone or email
  directly with each other, so `source1` acts as the bridge table linking
  them (a person must appear in source1 with both phone and email to connect
  a source2 record to a source3 record for that same person).

## Stack
- Python 3.10, pandas — Task 1
- n8n (local, no account required) — Task 2
- Plain HTML/JS, Web Audio API + MediaRecorder API — Task 3
