# Target Environment — Arduino UNO Q (Qualcomm QRB2210)

Captured from real hardware on **2026-05-27**. Source of truth:
`artifacts/probes/uno-q-2026-05-27.json`.

## Software baseline

| Property        | Value                                                              |
| --------------- | ------------------------------------------------------------------ |
| OS              | Debian GNU/Linux 13 (trixie)                                       |
| Kernel          | 6.16.7-g0dd6551ae96b — **mainline**, not Qualcomm downstream       |
| Architecture    | aarch64                                                            |
| libc            | GNU libc 2.41 (`Debian GLIBC 2.41-12+deb13u2`)                     |
| Default user    | `arduino` (uid 1000), member of `render`, `video`, `dialout`, `gpiod`, `audio` |
| Hostname        | `Mother`                                                           |
| Transport       | `adb` (Arduino's bundled binary at `~/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb`) |

## CPU

| Property        | Value                              |
| --------------- | ---------------------------------- |
| SoC             | QRB2210 (QCM2290 family)           |
| Cores           | 4× Qualcomm Kryo (Cortex-A53 derivative; impl `0x51`, part `0x801`) |
| Max freq        | 2.016 GHz per core                 |
| Min freq        | 300 MHz                            |
| Governor        | `schedutil`                        |
| Features        | `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid` |

ASIMD (NEON) is universally available across the cluster. No SVE. All four
cores are online and homogeneous.

## Memory

| Property        | Value                              |
| --------------- | ---------------------------------- |
| MemTotal        | ~1.78 GiB                          |
| MemAvailable    | ~1.06 GiB at capture time          |

LPDDR4X dual-channel per QRB2210 datasheet.

## Thermal

10 thermal zones exposed: `mapss`, `video`, `wlan`, `cpuss0`, `cpuss1`,
`mdm0`, `mdm1`, `gpu`, `hm-center`, `camera`. All idle in the 31–35 °C range
at capture time, so we have ample sustained-workload headroom (the QRB2210
throttles around 95 °C per Qualcomm spec).

## GPU

| Property        | Value                              |
| --------------- | ---------------------------------- |
| GPU             | Adreno A702 @ 845 MHz              |
| Kernel driver   | `msm_dpu` bound to the display controller; render node `/dev/dri/renderD128` |
| Devicetree compat | `qcom,qcm2290-dpu`               |
| KGSL legacy     | **absent** (`/dev/kgsl-3d0` missing — this is a mainline-kernel image) |
| Vulkan          | Mesa 25.2.6 **Turnip** driver, ICD `freedreno_icd.json`, API 1.0.318. Enumerates as `Turnip Adreno (TM) 702`, `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`, vendorID `0x5143` (Qualcomm), deviceID `0x07000200` |
| OpenCL          | **Rusticl** (`libRusticlOpenCL.so`) — Mesa/X.org platform, OpenCL 3.0, device `FD702`, vendor `Qualcomm`, 1 compute unit (CL view, not Adreno SP count), global mem ~1.74 GiB |
| LLVM softpipe   | `llvmpipe` Vulkan device also present as a CPU fallback (Mesa 25.2.6) |

## DSP

Hexagon DSP is **not exposed to userspace** on this image. No
`/dev/adsprpc-smd`, `/dev/cdsprpc-smd`, `/dev/fastrpc-*` nodes. Treat the DSP
as out of scope until either (a) Qualcomm's FastRPC drivers are upstreamed
to this image, or (b) we build an out-of-tree FastRPC module ourselves.

## STM32U585 link

| Property        | Value                              |
| --------------- | ---------------------------------- |
| Candidate TTYs  | `/dev/ttyGS0` (USB gadget serial), `/dev/ttyHS1` |
| Host USB        | No host-side USB devices enumerated (the A53 is itself the USB device in `adb` mode) |

The A53→STM32 bridge appears to be the USB gadget serial driver (`ttyGS0`).
The STM32U585 firmware itself is the concern of the sibling repository
`DragonWing-rs`; we just need a stable serial endpoint on the A53 side.

## Answered open questions

> Questions originally posed in `.task/implementation-task-001.md`.

1. **Is Mesa/Turnip already on the stock image?**
   **Yes.** Mesa 25.2.6 with Turnip is preinstalled at
   `/lib/aarch64-linux-gnu/libvulkan_freedreno.so`, ICD registered, and
   `vulkaninfo` enumerates the Adreno A702 as an integrated GPU under
   `DRIVER_ID_MESA_TURNIP`. We do not need to build Mesa from source for the
   workspace under `/Users/tarasworonjanski/Documents/LAB/PROJECTS/ARDUINO/DEV/mesa`.

2. **Is `/dev/kgsl-3d0` available without root?**
   **No — the node does not exist.** This image runs a mainline 6.16 kernel
   that uses the upstream `msm` DRM driver, not the Qualcomm downstream KGSL
   driver. The TinyGrad-style direct-KGSL-ioctl path (see
   `tinygrad/runtime/support/compiler_qcom.py`) cannot be applied as-is on
   this device. The supported access path is Vulkan compute (Turnip) and/or
   OpenCL (Rusticl), both of which sit on top of the upstream `msm` DRM
   driver under the hood.

3. **glibc version on-device?**
   **GNU libc 2.41 (Debian 13 trixie).** The `cross` project's default
   images target glibc 2.31/2.35 and would produce binaries that fail to
   resolve symbols at load time. We sidestepped the problem entirely by
   linking the probe binary statically against musl
   (`aarch64-unknown-linux-musl` + `rust-lld`). For any future binary that
   must link against system Vulkan/OpenCL ICDs, we will either use
   `aarch64-unknown-linux-gnu` with a Homebrew-installed
   `aarch64-unknown-linux-gnu-gcc` cross toolchain, or build inside a Debian
   trixie aarch64 container.
