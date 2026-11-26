- The warning for a `krishiv.memory.limit` above the memory the process may use is logged once per requested value,
  not on every resource-group admin check; the bench runner's `limit_bytes` is the memory pool's fitted limit; the
  setting's reference says that a value just under the effective limit leaves no headroom.
