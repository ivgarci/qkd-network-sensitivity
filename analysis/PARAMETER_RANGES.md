# Parameter ranges: literature grounding

Literature grounding for the seven uncertain inputs of `config.py`
(compiled 2026-09-17). Every range below carries a source and an endpoint
rationale, and records what the spread represents, so that the Sobol indices
can be interpreted correctly.

**Verification status legend**

- `VERIFIED-PRIMARY` — value read verbatim from the primary source text.
- `VERIFIED-RECORD` — bibliographic record confirmed at publisher or
  institutional repository.
- `SECONDARY` — value reported by a reliable secondary source; primary text not
  yet read in full.

---

## 1. Reference parameter set (anchor)

The canonical anchor for fibre decoy-state BB84 simulation is the GYS parameter
set, i.e. the experimental parameters of Gobby, Yuan and Shields (2004) as
tabulated by Ma, Qi, Zhao and Lo (2005), Table 1:

| Quantity | GYS value |
|---|---|
| Wavelength | 1550 nm |
| Fibre attenuation `alpha` | 0.21 dB/km |
| `e_detector` (misalignment) | 3.3 % |
| `Y_0` (background rate per pulse) | 1.7e-6 |
| `eta_Bob` (receiver efficiency) | 0.045 |
| Repetition rate | 2 MHz |

Status: `VERIFIED-PRIMARY` (arXiv:quant-ph/0503005, Table 1, read verbatim).

Ma et al. additionally state `f(e) = 1.22` as "the upper bound of `f(e)` in
secure distance for this experiment", and derive `mu_optimal ~ 0.48` for
`f(e) = 1.22` (`~ 0.54` for `f(e) = 1`), noting the key rate and distance are
"pretty stable against even a 20 % change of `mu`". Status: `VERIFIED-PRIMARY`.

> **Modelling note.** GYS `eta_Bob = 0.045` is the *total
> receiver* efficiency, including Bob's internal optical losses, not the bare
> detector efficiency. The manuscript's model writes `eta = eta_det * eta_ch`,
> so `eta_det` there occupies the role of `eta_Bob`. Either rename the symbol to
> `eta_Bob` / `eta_rx`, or state explicitly that internal receiver loss is folded
> into `eta_det`. Leaving this implicit invites a referee comment.

---

## 2. Recommended ranges

### 2.1 Fibre attenuation `alpha` [dB/km, 1550 nm]

**Recommended: 0.17 - 0.25, uniform.** (Manuscript proposed 0.16 - 0.25.)

| Endpoint | Value | Basis |
|---|---|---|
| Lower | 0.17 | Corning SMF-28 ULL specifies 0.164 dB/km average attenuation at 1550 nm *without splices*; 0.17 is the realistic spliced-span figure. `SECONDARY` |
| Anchor | 0.21 | GYS experimental value. `VERIFIED-PRIMARY` |
| Upper | 0.25 | G.652.D vendor specification figure at 1550 nm; represents standard deployed fibre including ageing and splice budget. `SECONDARY` |

Rationale for the nudge from 0.16 to 0.17: 0.164 dB/km is an unspliced
laboratory-grade figure for a premium ULL product. Using it as the lower bound
of a *deployed* network model overstates the best case. Ultra-low-loss fibre is
used in long-distance QKD demonstrations (Stucki et al. 2009, 250 km;
Boaron et al. 2018, 421 km), so the ULL end of the range is legitimate - but
0.17 is the honest spliced value.

> **DA Checkpoint 2 - substantive challenge.** In the MadQCI deployment (the
> Spanish production QKD network, Telefonica + RedIMadrid), link lengths of
> 1.9 - 33 km carry 2.0 - 14.3 dB of C-band loss. That is an *effective* per-km
> loss well above any fibre-intrinsic `alpha`, because connectors, patch panels
> and splices contribute a largely distance-independent insertion loss.
> A model that maps distance to loss through `alpha * d` alone will therefore be
> optimistic for exactly the short metropolitan links that dominate the
> Castilla y Leon scenario. **Recommendation:** add a fixed insertion-loss term
> `L_0` (dB) to the channel model, or widen `alpha` and document that it is an
> *effective* coefficient rather than a fibre property. This is a modelling
> decision, not a range decision - it belongs to the model definition. It is flagged here so
> it is not lost.

### 2.2 Detector / receiver efficiency `eta_det` [-]

**Recommended: 0.10 - 0.85, uniform.** (As proposed. No change.)

| Endpoint | Value | Basis |
|---|---|---|
| Lower | 0.10 | Commercial gated InGaAs/InP SPADs operate around 10 - 20 % detection efficiency; gated implementations reported at 10.8 % and 11 %. `SECONDARY` |
| Mid | 0.40 - 0.60 | Research-grade InGaAs/InP: 40 % (2.3 kcps dark counts), 60 % reported. `SECONDARY` |
| Upper | 0.85 | Practical SNSPD *system* detection efficiency. NIST reports 98.0 +/- 0.5 % as a laboratory record and >90 % is achieved at cryocooler temperature; 0.85 represents a deployable system rather than a record. `SECONDARY` |

The range deliberately spans two detector technologies. This is the source of
the independence artefact flagged at Checkpoint 1 - see section 4.

### 2.3 Dark-count probability `p_dc` [per gate]

**Recommended: 1e-8 - 1e-5, log-uniform.** (As proposed. No change.)

| Endpoint | Value | Basis |
|---|---|---|
| Lower | 1e-8 | SNSPD dark count rates reach Hz and below (0.1 Hz in Boaron et al. 2018; sub-Hz with cold optical filtering). At ns-scale gating this is `<= 1e-8` per gate. `SECONDARY` |
| Anchor | ~8.5e-7 | Implied by GYS `Y_0 = 1.7e-6` under the two-detector relation `Y_0 ~ 2 p_dc` used in the manuscript. `VERIFIED-PRIMARY` (for `Y_0`) |
| Mid | 6.3e-7, 7e-7 | Directly reported per-gate dark-count probabilities for sinusoidally / rectangularly gated InGaAs/InP SPADs. `SECONDARY` |
| Upper | 1e-5 | Warm or high-efficiency InGaAs operation (kcps-range dark count rates over longer effective gates). `SECONDARY` |

Log-uniform sampling is correct and should be retained: the range spans three
decades and the physics is multiplicative.

### 2.4 Optical misalignment error `e_d` [-]

**Recommended: 0.005 - 0.040, uniform.** (As proposed. No change.)

| Endpoint | Value | Basis |
|---|---|---|
| Lower | 0.005 | Phase-encoded QKD achieving QBER < 1 % out to 225 km in standard telecom fibre; the misalignment floor is the sub-1 % part of that budget. `SECONDARY` |
| Common assumption | 0.015 | 1.5 % is a widely used constant in theoretical treatments. `SECONDARY` |
| Upper | 0.033 - 0.040 | GYS `e_detector = 3.3 %` (`VERIFIED-PRIMARY`); deployed polarisation-encoded field trials report QBER 3.2 - 5.0 %. `SECONDARY` |

Ma et al. note `e_detector` is "more or less independent of the distance" even
at 122 km, which justifies treating it as a distance-independent constant in the
link model - a modelling assumption the manuscript makes implicitly and should
state explicitly with this citation.

### 2.5 Signal-state intensity `mu` [photons/pulse]

**Recommended: 0.20 - 0.80, uniform.** (As proposed. No change.)

Justification: the optimum sits at `mu ~ 0.48` for `f_EC = 1.22`
(`VERIFIED-PRIMARY`), and the practical literature places signal intensities at
0.4 - 0.7. The proposed range is deliberately *wider* than the optimum on both
sides. That is methodologically correct for a GSA: the point is to measure how
much variance protocol mis-tuning contributes, which requires sampling away
from the optimum. State this rationale in Section 3.3 so it is not mistaken for
a loose prior.

### 2.6 Error-correction inefficiency `f_EC` [-]

**Recommended: 1.05 - 1.22, uniform.** (As proposed. No change.)

| Endpoint | Value | Basis |
|---|---|---|
| Lower | 1.05 | Modern LDPC reconciliation approaches the Shannon limit arbitrarily closely. `SECONDARY` |
| Mid | 1.10 - 1.20 | Cascade achieves ~1.1 - 1.2 across QBER 0 - 11 %. `SECONDARY` |
| Upper | 1.22 | Used by Ma et al. as the upper bound of `f(e)` over the secure distance. `VERIFIED-PRIMARY` |

Experimental implementations at `f = 1.17` and `f = 1.25` bracket this range,
so 1.22 is a defensible - not conservative - upper bound.

---

## 3. Sources

Primary, read verbatim:

- Ma, X.; Qi, B.; Zhao, Y.; Lo, H.-K. Practical decoy state for quantum key
  distribution. *Phys. Rev. A* **2005**, *72*, 012326. arXiv:quant-ph/0503005.
  (Table 1 - GYS parameter set.)

Bibliographic records confirmed:

- Mehic, M. et al. Quantum Key Distribution: A Networking Perspective.
  *ACM Comput. Surv.* **2020**, *53*(5), Article 96. DOI 10.1145/3402192.
- Cao, Y. et al. The Evolution of Quantum Key Distribution Networks: On the Road
  to the Qinternet. *IEEE Commun. Surv. Tutor.* **2022**, *24*(2), 839-894.
  DOI 10.1109/COMST.2022.3144219.

Newly identified, to add to `references.bib`:

- Gobby, C.; Yuan, Z.L.; Shields, A.J. Quantum key distribution over 122 km of
  standard telecom fiber. *Appl. Phys. Lett.* **2004**, *84*, 3762-3764.
- Stucki, D. et al. High rate, long-distance quantum key distribution over
  250 km of ultra low loss fibres. *New J. Phys.* **2009**, *11*, 075003.
  DOI 10.1088/1367-2630/11/7/075003.
- Boaron, A. et al. Secure quantum key distribution over 421 km of optical
  fiber. *Phys. Rev. Lett.* **2018**, *121*, 190502. arXiv:1807.03222.
- Martin, V. et al. MadQCI: a heterogeneous and scalable SDN-QKD network
  deployed in production facilities. *npj Quantum Inf.* **2024**.
  DOI 10.1038/s41534-024-00873-2.
- ITU-T Recommendation G.652, Characteristics of a single-mode optical fibre
  and cable, edition 08/2024.

---

