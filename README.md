# Clean Before Closed

Code for *Clean Before Closed: How Much of the Health Benefit of Retiring Coal Plants Arrives
Before Retirement* (Arnay Mukhtar, working paper, November 2026).

**Paper:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7690098  
**Author:** Arnay Mukhtar, Independent Researcher, Almaty, Kazakhstan (ORCID [0009-0003-5546-0418](https://orcid.org/0009-0003-5546-0418))  
**Pre-registration:** [10.17605/OSF.IO/V7S6E](https://doi.org/10.17605/OSF.IO/V7S6E), filed 9 November 2026; addendum for the Almaty prediction [10.17605/OSF.IO/D5Q34](https://doi.org/10.17605/OSF.IO/D5Q34)  
**Code and data archive:** OSF project [osf.io/e3unc](https://osf.io/e3unc), DOI [10.17605/OSF.IO/E3UNC](https://doi.org/10.17605/OSF.IO/E3UNC), with SHA-256 fingerprints for every file

---

## What the paper finds

Across 878 US coal unit exits from 2005 to 2024, 80% of the fall in sulfur dioxide from each
unit's peak had already happened by its last full year of operation, mostly before any
recorded plan to retire; two thirds of the decade's decline came from units running less.
Against a validated counterfactual of matched units, crediting the last full year overstates
the benefit of retirement 1.7 times over five years and twice over ten; crediting emissions
from ten years earlier overstates it four to five times.

The causal half was pre-registered before any air-quality data near an exit was viewed. At 51
large exits it rules out lasting local PM2.5 falls of 1.1 µg/m³, the published closure-month
size, while sulfate falls by an amount consistent with a reduced-complexity model (InMAP).

---

## Repository layout

```
scripts/                          every analysis script, as deposited on OSF
outcomes/                         the outputs those scripts write (tables, event paths, test results)
post_registration_fingerprints.csv  SHA-256 of every deposited script, output and data file
```

The input data are not duplicated here; they are on the OSF project (folders `data/` and
`post_registration/data/`). All of them come from public sources: EPA CAMPD emissions,
EIA-860, EPA AQS monitors, the InMAP Source-Receptor Matrix, MAIAC aerosol and TROPOMI NO2
(through Google Earth Engine), EEA and ENTSO-E for Europe, and CEA and Eskom for India and
South Africa.

---

## Which script does what

Scripts are listed in the order the work was done; a few small checking scripts are left out of the table. The registration was filed after the
design stage and before any script in the "registered tests" group was run.

| Stage | Scripts | Paper |
|---|---|---|
| Exits and emissions | `usunits.py`, `usevents.py`, `annualcheck.py`, `rerunall.py`, `emisdiag.py`, `regime.py` | Sections 3.1–3.2, 4.1 |
| Modelled effects and valuation | `isrminspect.py`, `isrmextract.py`, `isrmrows.py`, `isrmpredict.py`, `valuepred.py`, `val2010.py`, `easiur.py` | Sections 3.3, 4.2 |
| Design and power, before filing (no outcomes near exits read) | `aqscount.py`, `aqspower.py`, `step1pm.py`, `step1diag.py`, `step2so4.py`, `step3wind.py`, `steps345.py`, `hourlycov.py`, `maiacstart.py`, `maiacpower.py`, `prefiling.py`, `tune.py`, `finalize.py` | Section 5 |
| Registered tests, after filing | `outcomes1.py`, `outcomes2.py` (PM2.5 and sulfate), `maiacevents.py`, `outcomes3.py` (satellite aerosol, H3f) | Sections 6.1–6.6 |
| Counterfactual benchmark, replacement, which plants run to the end | `outcomes4.py`, `cfrobust.py`, `netrepl.py`, `netrepl2.py`, `batchA.py` | Sections 4.3–4.6 |
| Registered secondary designs | `batchD.py`, `batchE.py`, `batchF.py` | Section 6.9 |
| Exploratory: event-time paths, published closure designs, robustness | `eventtime.py`, `batchB.py`, `btdesign.py`, `batchC.py` | Sections 6.7–6.8, Online Appendix B |
| Other countries (descriptive) | `eeaclosures.py`, `eealcp.py`, `eeaunits.py`, `next2.py` (Europe), `q5cea.py`, `q5india.py` (India) | Sections 4.1, 7.2 |
| Almaty CHP-2 prediction | `almatypred.py` | Addendum D5Q34 |
| Figures | `figmap.py` (Figure 4), `fig9.py` (Figure 9) | |
| Deposit | `osfupload.py`, `osfupload2.py` | |

Each script opens with a comment block saying what it reads, what it writes and whether it ran
before or after the registration was filed. Deviations from the registered plan are listed in
Table 4 of the paper.

---

## Running the code

The scripts were run in Google Colab with the data in a Google Drive folder
`MyDrive/coal_closures` (each script sets `DIR` to that path). To rerun:

1. Download the `data/` and `post_registration/data/` folders from the OSF project into that
   Drive folder.
2. Open a script in Colab and run it. Scripts that call web services read their keys from
   Colab Secrets (for example `EPA_KEY`, `ENTSOE_KEY`); no key is stored in any file. The
   satellite scripts need a Google Earth Engine account.
3. Compare outputs against `outcomes/` and the SHA-256 values in
   `post_registration_fingerprints.csv`.

---

## Licence

Code released under the MIT Licence. The input data remain subject to their providers' terms.

## Citation

Mukhtar, A. (2026). *Clean Before Closed: How Much of the Health Benefit of Retiring Coal Plants
Arrives Before Retirement.* Working paper, SSRN 7690098.
