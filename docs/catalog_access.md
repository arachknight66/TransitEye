# Catalog access contract

TransitEye retrieves the NASA Exoplanet Archive TAP table `toi` through its
synchronous HTTPS endpoint. The query explicitly selects, in fixed order:

`tid`, `toi`, `tfopwg_disp`, `pl_orbper`, `pl_tranmid`, `pl_trandurh`,
`pl_trandep`, `toi_created`, and `rowupdate`.

The field names were verified against the Archive's TOI column documentation
on 2026-09-09. They represent TIC identity, TOI identity, TFOPWG disposition,
orbital period, transit midpoint, transit duration, transit depth, creation
date, and last update date respectively.

The implementation never uses `SELECT *`. A response missing any requested
field, a malformed TOI identifier, or a non-numeric non-null ephemeris value
is rejected with an explicit error. Missing TIC or ephemeris values are kept as
explicit missing values rather than fabricated or silently dropped.

Each successful retrieval is stored as an immutable snapshot with:

- exact raw CSV response;
- normalized Parquet table;
- metadata containing query, endpoint, UTC time, row count, and checksums.

The catalog is external mutable state. Later stages must reference a frozen
snapshot ID and may not re-query live state as a substitute.
