Group-by and COUNT(DISTINCT) over one or two fixed-width keys without NULLs encode their rows in one pass: ClickBench
q09 uses 0.94x the CPU cycles and 0.83x the instructions, q10 and q16 0.86x and 0.89x the instructions (scale factor 10).
