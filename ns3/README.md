# ns-3 scenarios

Three ns-3.42 programs produce every measurement in the paper. They are
distributed under GPL-2.0-only (see `LICENSE`); `qtro-closedloop.cc` contains
a queue disc derived from ns-3's `RedQueueDisc` and keeps that file's notices.

| program | used by |
|---|---|
| `qtro-dumbbell.cc` | all stationary datasets, the burst shift, the load steps, the cost timing |
| `qtro-multibottleneck.cc` | the tandem shift dataset |
| `qtro-closedloop.cc` | the closed loop and the two state probes |

## Build

```sh
cd /path/to/ns-allinone-3.42/ns-3.42
cp /path/to/TCP_QNO/ns3/qtro-*.cc scratch/
./ns3 configure --build-profile=optimized --enable-examples
./ns3 build
ls build/scratch/ns3.42-qtro-*-optimized      # the three binaries
```

The Python scripts find the tree through `NS3_DIR` (or `--ns3-dir`) and start
it in one of two modes:

- `native` (default): `bash -lc "cd $NS3_DIR && ./ns3 run ..."` on this
  machine.
- `wsl`: the same command inside a WSL distribution, `NS3_DISTRO` (default
  `Ubuntu-24.04`), optionally as the user who owns the build, `NS3_USER`,
  entered with `su`. This is how the published runs were made from Windows.

```sh
export NS3_DIR=/opt/ns-allinone-3.42/ns-3.42   # native
export NS3_MODE=wsl NS3_USER=ns3               # or from Windows through WSL
```

From Git Bash on Windows, set `MSYS_NO_PATHCONV=1` as well, or pass
`--ns3-dir` explicitly: MSYS otherwise rewrites `/opt/...` into a path under
the Git installation before Python sees it.

The open-loop scenarios run through the `./ns3 run` wrapper. The closed loop
calls the optimised binary directly, because the wrapper does not forward
stdin and the control channel needs it; the state probes use the same binary
with stdin closed.

Each program prints one CSV line per sample; the closed loop additionally
prints `CTRL,...` lines and reads `<pMax>,<minTh>` replies on stdin. The flag
names are listed at the top of each file.
