# Data guide

All four cities and all twelve split files supplied in HLFT.zip are included as gzip-compressed text. The loader reads .txt.gz directly; no manual decompression is needed. Compression does not alter observations. Uncompressed-file hashes and sizes are recorded in `data/manifest.json`.

Every row has four fields:

```text
sensor_id::time_interval_id::day_id::speed
```

The filenames retain the original split labels:

| City | Training | Validation | Test |
| --- | --- | --- | --- |
| Berlin | Berlin_train70.txt.gz | Berlin_valid10.txt.gz | Berlin_test20.txt.gz |
| Guangzhou | Guangzhou_train70.txt.gz | Guangzhou_valid10.txt.gz | Guangzhou_test20.txt.gz |
| NewYork | NewYork_train70.txt.gz | NewYork_valid10.txt.gz | NewYork_test20.txt.gz |
| Seattle | Seattle_train70.txt.gz | Seattle_valid10.txt.gz | Seattle_test20.txt.gz |

Each city has a folder under `data/`. Missing tensor entries are implicit; a zero-valued measurement and an unobserved entry are not generally the same thing. For continuity with the supplied source, the default omits zero-valued observations. New York contains substantial zero-valued observations, so this choice changes its evaluated population. The audit table and run metadata disclose raw and effective counts. Use `--keep-zero` for an explicitly separate experiment if zeros represent valid stopped traffic in your intended analysis.

The split audit found no duplicate coordinates within a file and no coordinate overlap between train, validation, and test in any city. Splits were supplied already prepared; the original split-generation seed and the full upstream preprocessing recipe are not present in this archive. No new claim about those preprocessing steps is made.

Upstream sources cited in the supplied published paper:

- Berlin: https://daten.berlin.de
- Guangzhou: https://zenodo.org/record/1205229
- Seattle: https://github.com/zhiyongc/Seattle-Loop-Data
- New York: https://opendata.cityofnewyork.us/

Dataset credit remains with the original providers. The included files are the research splits supplied with the source, not fresh downloads from those portals. No ownership of the upstream datasets is asserted.
