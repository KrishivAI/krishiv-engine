`krishiv server` (Spark Connect) no longer fails the ML cache cleanup PySpark sends when a client process exits.
An operation interrupted or released while its plan is analysed ends with `OPERATION_CANCELED`, an internal failure
while results are encoded ends the stream with `INTERNAL_ERROR` instead of leaving it waiting, and a released
operation is no longer reported as abandoned.
