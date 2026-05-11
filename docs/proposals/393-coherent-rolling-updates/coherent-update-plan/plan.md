# Plan: Coherent Rolling Updates — Minimum Viable Implementation

## Context

Disaggregated inference stacks (prefill + decode pods) require that pods updating together are **version-compatible** — cross-version communication causes silent correctness failures. The existing `RollingRecreate` strategy updates replicas one-at-a-time per PCS replica but does not guarantee that all pods forming a minimum-viable serving unit are updated atomically and gang-scheduled together.

This plan implements the `Coherent` update strategy from GREP-393 (minus revision history / rollback, which are out of scope). Coherent updates replace pods in **Minimal Viable Units (MVUs)** — `MinAvailable` replicas of each updated standalone PCLQ plus `MinAvailable` PCSG replicas per updated PCSG — scheduled atomically via new **MVUPodGangs (MPGs)**.

---

## RollingRecreate vs Coherent: Keep Both or Disable?

### Entanglement analysis

After thorough code review, the shared surface between the two strategies is **narrow and well-bounded**:

| Shared element | Used by both? | Notes |
|---|---|---|
| `initUpdateProgress()` in PCS `reconcilespec.go:138` | Yes | Sets `UpdateStartedAt`; generic |
| `isAutoUpdateInProgress()` in `podcliquesetreplica.go:307` | Yes — but Coherent needs its own check | Rename + split |
| PCLQ `processUpdate()` | Yes — but Coherent needs early-exit | Add 3-line guard |
| PCSG `processUpdate()` | Yes — but Coherent needs early-exit | Add 3-line guard |
| PCLQ/PCSG/PCS `reconcilestatus.go` | Yes | Fully generic — no changes needed |
| `getPCSReplicaDeletionWork()` (gang termination) | Yes | Fully generic — no changes needed |

The **RollingRecreate path does not need to be touched** to implement Coherent. The two strategies diverge cleanly at:
- `podcliquesetreplica.go:Sync()` — dispatch by strategy type
- `PCLQ/PCSG processUpdate()` — early-exit guard for Coherent

**Recommendation: Keep both strategies.** The additional complexity is 3-5 lines of guards. Disabling RollingRecreate would require removing working code and re-adding it later. The implementation below keeps it intact.

---

## PodGang Model: Unified MPG-based naming

### Old model (current)
Two fixed PodGang types exist today:
- **BPG (BasePodGang):** one per PCS replica, named `<pcs-name>-<pcs-replica-index>`, contains all standalone PCLQs + first `MinAvailable` PCSG replicas
- **SPG (ScaledPodGang):** one per PCSG replica above `MinAvailable`, named `<pcsg-fqn>-<scaled-index>`

### New model (this implementation)
A single unified PodGang type — **MPG** — computed from the MVU template, named:
```
<pcs-name>-<pcs-replica-index>-<short-generationhash>-<createdPodGangCount-1>
```

Where:
- `short-generationhash` is a truncated prefix (5 chars) of `pcs.Status.CurrentGenerationHash` — encodes *which update* the MPG belongs to; 5 chars provides sufficient collision resistance within a single PCS's update history (27^5 ≈ 14.3M possible values)
- `createdPodGangCount-1` is `PodGangState[replicaIndex].CreatedPodGangCount - 1` at the time the PodGang is created — the count is incremented after creation, so the name uses the pre-increment value

This avoids any global monotonic growth. Names are meaningful (you can tell from the name which update event and which MVU iteration created the PodGang), and the iteration count is bounded by `ceil(max(old_standalone_pclq_pods / MinAvailable_pclq, old_pcsg_replicas / MinAvailable_pcsg))` — a small number tied to the PCS replica count, not something that grows over the lifetime of the PCS.

### Unification rules

MPG composition depends on which PCLQs/PCSGs are actually updated (the "update scope"), following the rules in GREP-393 § "Rules of MVU gang-scheduling":

**Standalone PCLQs:**
- If only one standalone PCLQ is updated and its `minAvailable == 1` → **no MPG created**; new pods for that PCLQ are placed back into the original PG (BPG or existing MPG).
- If only one standalone PCLQ is updated and its `minAvailable > 1` → one or more MPGs, each containing exactly `minAvailable` replicas of that PCLQ. Remaining replicas fill subsequent MPGs `minAvailable` at a time; the final batch (if less than `minAvailable`) is appended to the last MPG.
- If more than one standalone PCLQ is updated → one or more MPGs, each containing exactly `minAvailable` replicas of each updated standalone PCLQ. Remaining replicas of each PCLQ fill subsequent MPGs `minAvailable` at a time; leftover replicas that don't fill a complete MVU are appended to the last MPG.

**PCSGs:**
- If one or more PCLQs of a PCSG are updated, the entire PCSG is treated as updated. Each MPG contains `minAvailable` PCSG replicas; all constituent PCLQs (with all their replicas) of each PCSG replica are included as PodGroups.
- Multiple updated PCSGs: each MPG includes `minAvailable` replicas of each updated PCSG.
- Mix of updated standalone PCLQs and PCSGs: each MPG includes both.

**Tail-MPGs:** When the remaining old pods/replicas in the take-down set don't fill a complete MVU (i.e. the tail), they form one or more Tail-MPGs — each containing one PCSG replica's worth of pods. Tail-MPG gates are removed only after the preceding non-tail MPG becomes available.

- **New PCS (initial deployment under Coherent):** creates MPGs directly using the above composition rules. No BPGs/SPGs ever created.
- **Existing PCS migrating to Coherent:** old BPGs/SPGs lose pod references as pods move into new MPGs during the update. Once empty, they are deleted by the existing excess-detection logic in `getExcessPodGangNames()` — no forced deletion, no scheduler disruption.

### Counter storage

`CreatedPodGangCount` is a **per-replica, always-present** counter stored in `PodCliqueSetStatus` — outside the update boundary so it is available for scale-out events even when no update is in progress:

```go
// PodGangReplicaState tracks the persistent PodGang creation counter for a single PCS replica.
// Always present (one entry per replica); survives across updates and scale-out events.
type PodGangReplicaState struct {
    ReplicaIndex        int32 `json:"replicaIndex"`
    // CreatedPodGangCount is the total number of PodGangs ever created for this replica.
    // Used to derive the next PodGang name. Never resets — monotonically increasing.
    CreatedPodGangCount int32 `json:"createdPodGangCount"`
}
```

Stored in `PodCliqueSetStatus` as:
```go
PodGangState []PodGangReplicaState `json:"podGangState,omitempty"`
```

`CoherentReplicaUpdateProgress` no longer carries `CreatedPodGangCount` — it reads/increments from `PodGangState` directly:

```go
type CoherentReplicaUpdateProgress struct {
    ReplicaIndex    int32        `json:"replicaIndex"`
    UpdateStartedAt metav1.Time  `json:"updateStartedAt"`
    UpdateEndedAt  *metav1.Time `json:"updateEndedAt,omitempty"`
    // InFlightPodGangs are the names of all PodGangs for the current round (one non-tail MPG +
    // zero or more tail-MPGs). The orchestrator waits for all of them to become available before
    // proceeding to the next round. They may not yet exist or may exist but not yet be available.
    // Purely observability — PodGangMap is the source of truth for pod/PodGang assignment.
    InFlightPodGangs []string `json:"inFlightPodGangs,omitempty"`
    // ErrorMessage captures the reason the update of this replica is failing or stalled, if any.
    ErrorMessage *string `json:"errorMessage,omitempty"`
}
```

The counter never resets — it increments monotonically across updates and scale-out events, guaranteeing unique PodGang names for a given replica over the lifetime of the PCS. The `short-generationhash` segment in the name provides additional scoping per update event.

---

## PodGangMap CRD

`PodGangMap` is a new desired-state CRD that serves as the **single source of truth** for the mapping between PodGangs and PCLQ/PCSG pod counts for a given PCS replica. It is created and reconciled by the `PodGangMap` component in the PCS reconciler before any PG, PCLQ, or PCSG components run. All other components (PodGang, PCLQ, PCSG) read it and act on it — no component writes to it except the `PodGangMap` component itself.

One `PodGangMap` resource per PCS replica, named `<pcs-name>-<pcs-replica-index>`.

```go
type PodGangMapSpec struct {
    PodCliqueSetReplicaIndex int32          `json:"podCliqueSetReplicaIndex"`
    Tuples                   []PodGangTuple `json:"tuples"`
}

type PodGangTuple struct {
    // Name is the PodGang name this tuple corresponds to.
    Name string `json:"name"`
    // PodCliqueSetGenerationHash is the PCS generation hash of pods that should belong to this tuple.
    // Used by PCLQ/PCSG reconcilers to create pods with the correct spec version.
    PodCliqueSetGenerationHash string `json:"podCliqueSetGenerationHash"`
    // PodCliques maps standalone PCLQ name to the number of pods that belong to this PodGang.
    PodCliques map[string]int `json:"podCliques,omitempty"`
    // PodCliqueScalingGroups maps PCSG name to the number of PCSG replicas that belong to this PodGang.
    PodCliqueScalingGroups map[string]int `json:"podCliqueScalingGroups,omitempty"`
}
```

### PodGangMap computation

On every reconcile, the `PodGangMap` component computes the desired tuples from:
- `pcs.Spec` — structural template (which PCLQs, which PCSGs, `MinAvailable` values)
- Live PCLQ/PCSG resources — actual current replica counts (reflecting any scale-out/in)
- `pcs.Status.CoherentUpdateProgress` — which replica is being updated, which MPGs exist so far

#### Case 1: Existing PCS (BPG/SPG topology, no update in progress)

Tuples derived purely from `pcs.Spec` + live PCLQ/PCSG resources matching BPG/SPG convention:
- One tuple named `<pcs-name>-<replica>` (BPG): `{podCliqueSetGenerationHash: current, all standalone PCLQs at full replica count, all PCSGs at MinAvailable replicas}`
- One tuple per PCSG replica above `MinAvailable` named `<pcsg-fqn>-<scaled-index>` (SPG): `{podCliqueSetGenerationHash: current, that PCSG: 1}`

#### Case 2: New PCS (MPG topology, no update in progress)

Tuples computed from `pcs.Spec` + live PCLQ/PCSG resources using MVU composition rules:
- One tuple per MPG named via `GeneratePodGangName`, composition per the Unification rules section

#### Case 3: Coherent update in progress

Tuples reflect the partially-updated state:
- Old BPG/SPG tuples with decremented counts (remaining old pods not yet taken down), `podCliqueSetGenerationHash: old`
- Already-created MPG tuples from previous iterations, `podCliqueSetGenerationHash: new`
- Current iteration MPG tuple (from `InFlightPodGangs`), `podCliqueSetGenerationHash: new`
- Tail-MPG tuples if applicable

On each reconcile, counts are recomputed from live PCLQ/PCSG resources + `CoherentUpdateProgress` — a concurrent scale-out/in is automatically reflected on the next reconcile.

#### Case 4: RollingRecreate update in progress

Tuples remain structurally identical to steady-state (same PodGang names, same counts). Only `PodCliqueSetGenerationHash` is updated to the new PCS generation hash when the update is initiated. PCLQ/PCSG reconcilers see the hash change and replace pods in-place within the same PodGang.

### How PCLQ reconciler uses PodGangMap

For each tuple referencing this PCLQ:
1. Count existing pods that have matching `PodCliqueSetGenerationHash` AND `grove.io/podgang` label matching this tuple's `Name`
2. Create delta pods (up to the tuple's count) with `grove.io/podgang: <tupleName>` and scheduling gate set at creation time — no post-creation patching needed
3. If a pod is evicted and the tuple's quota is already satisfied by surviving pods → do not create a replacement

This eliminates the 1:N label-patching problem: `grove.io/podgang` is set correctly at pod creation time, not patched afterwards.

### How PodGang component uses PodGangMap

`computeExpectedPodGangs` reads `PodGangMap` tuples directly — one PodGang per tuple name. No name derivation logic needed in the sync flow itself. `getExcessPodGangNames()` is unchanged — old BPGs/SPGs become excess once their tuple is removed from `PodGangMap` (i.e. all their pods have moved to MPG tuples).

---

## Implementation Steps

### Step 1 — API changes

#### 1a — PCS API (`operator/api/core/v1alpha1/podcliqueset.go`)

1. Add `CoherentStrategy UpdateStrategyType = "Coherent"` constant.
2. Update kubebuilder enum validation: `+kubebuilder:validation:Enum={RollingRecreate,Coherent,OnDelete}`.
3. Add to `PodCliqueSetStatus`:
   - `CoherentUpdateProgress *CoherentUpdateProgress`
   - `PodGangState []PodGangReplicaState` — always-present per-replica PodGang counter
4. Define new types:
```go
// PodGangReplicaState tracks the persistent PodGang creation counter for a single PCS replica.
type PodGangReplicaState struct {
    ReplicaIndex        int32 `json:"replicaIndex"`
    // CreatedPodGangCount is the total number of PodGangs ever created for this replica.
    // Used to derive the next PodGang name. Never resets — monotonically increasing.
    CreatedPodGangCount int32 `json:"createdPodGangCount"`
}

type CoherentUpdateProgress struct {
    UpdateStartedAt   metav1.Time                     `json:"updateStartedAt"`
    UpdateEndedAt    *metav1.Time                    `json:"updateEndedAt,omitempty"`
    CurrentlyUpdating []CoherentReplicaUpdateProgress `json:"currentlyUpdating,omitempty"`
}

type CoherentReplicaUpdateProgress struct {
    ReplicaIndex    int32        `json:"replicaIndex"`
    UpdateStartedAt metav1.Time  `json:"updateStartedAt"`
    UpdateEndedAt  *metav1.Time `json:"updateEndedAt,omitempty"`
    // InFlightPodGangs are the names of all PodGangs for the current round (one non-tail MPG +
    // zero or more tail-MPGs). The orchestrator waits for all of them to become available before
    // proceeding to the next round. They may not yet exist or may exist but not yet be available.
    // Purely observability — PodGangMap is the source of truth for pod/PodGang assignment.
    InFlightPodGangs []string `json:"inFlightPodGangs,omitempty"`
    // ErrorMessage captures the reason the update of this replica is failing or stalled, if any.
    ErrorMessage *string `json:"errorMessage,omitempty"`
}
```

#### 1b — PodGangMap CRD (`operator/api/core/v1alpha1/podgangmap.go`) — new file

```go
type PodGangMap struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec   PodGangMapSpec   `json:"spec,omitempty"`
}

type PodGangMapSpec struct {
    PodCliqueSetReplicaIndex int32          `json:"podCliqueSetReplicaIndex"`
    Tuples                   []PodGangTuple `json:"tuples"`
}

type PodGangTuple struct {
    Name                   string         `json:"name"`
    PodCliqueSetGenerationHash string         `json:"podCliqueSetGenerationHash"`
    PodCliques             map[string]int `json:"podCliques,omitempty"`
    PodCliqueScalingGroups map[string]int `json:"podCliqueScalingGroups,omitempty"`
}
```

5. Run `make generate` to regenerate deepcopy and CRD manifests.

**Files:** `operator/api/core/v1alpha1/podcliqueset.go`, `operator/api/core/v1alpha1/podgangmap.go`

---

### Step 2 — PodGang naming: add unified MPG naming function (`operator/api/common/namegen.go`)

1. **Keep** `GenerateBasePodGangName()`, `CreatePodGangNameFromPCSGFQN()`, `GeneratePodGangNameForPodCliqueOwnedByPodCliqueSet()`, `GeneratePodGangNameForPodCliqueOwnedByPCSG()` — still used for BPG/SPG topology (existing PCS, no active update).
2. **Add** `GeneratePodGangName(pcsName string, replicaIndex int, shortGenerationHash string, createdPodGangCount int) string` — returns `<pcsName>-<replicaIndex>-<shortGenerationHash>-<createdPodGangCount-1>`.
   - Caller passes the current `PodGangState[replicaIndex].CreatedPodGangCount` value (pre-increment); function subtracts 1 to form the name suffix
   - `CurrentGenerationHash` must always be set (on creation and every spec change) so this function is always callable

**Files:** `operator/api/common/namegen.go`, `operator/api/common/namegen_test.go`

---

### Step 3 — New PodGangMap component (`operator/internal/controller/podcliqueset/components/podgangmap/`)

New component responsible for computing and reconciling `PodGangMap` resources. Runs **before** PodGang, PCLQ, and PCSG components in the PCS reconcile loop.

#### `computeTuples(pcs, existingPCLQs, existingPCSGs) []PodGangTuple`

Computes desired tuples based on PCS state:

- **Case 1 (existing BPG/SPG, no update):** derive BPG/SPG-convention tuples from `pcs.Spec` + live PCLQ/PCSG replica counts
- **Case 2 (MPG topology, no update):** derive MPG-convention tuples from `pcs.Spec` + live PCLQ/PCSG replica counts using MVU composition rules
- **Case 3 (Coherent update in progress):** old tuples with decremented counts + new MPG tuples from `InFlightPodGangs` + current iteration MPG tuple
- **Case 4 (RollingRecreate update):** same structure as steady-state, `PodCliqueSetGenerationHash` updated to new PCS generation hash

The component creates/updates the `PodGangMap` resource for each PCS replica. On scale-out, a new `PodGangMap` is created for the new replica. On scale-in, the `PodGangMap` for the removed replica is deleted.

#### Desired state semantics per scenario

- **RollingRecreate update:** `PodGangMap` always reflects the **complete desired state** — the full set of PodGangs that should exist with the new `PodCliqueSetGenerationHash`. PCLQ/PCSG reconcilers replace pods in-place within the same PodGangs.
- **Regular reconcile (no update, includes scale-out/in):** `PodGangMap` reflects the **complete desired state** — adjusts pod counts per PCLQ/PCSG in each existing tuple to reflect current replica counts, and adds/removes tuples for new/removed replicas. Always self-consistent.
- **Coherent update:** `PodGangMap` reflects only the **desired state for the current iteration** — one MVU at a time. It is not a complete desired end-state but a stepwise approximation that advances each time the orchestrator moves to the next iteration. Old tuples are decremented as pods are taken down; new MPG tuples are added one round at a time.

This distinction is important: consumers of `PodGangMap` (PodGang, PCLQ, PCSG components) always treat it as the authoritative desired state, but under Coherent updates they should expect it to change each iteration rather than converging in a single reconcile.

**Files:** `operator/internal/controller/podcliqueset/components/podgangmap/podgangmap.go` (new)

---

### Step 4 — Rewrite PodGang sync flow (`operator/internal/controller/podcliqueset/components/podgang/syncflow.go`)

Replace BPG+SPG computation with `PodGangMap`-driven computation:

**Remove:** `buildExpectedBasePodGangForPCSReplicas()`, `buildExpectedBasePodGangForPCSReplica()`, `buildExpectedScaledPodGangsForPCSG()`, `doBuildExpectedScaledPodGangForPCSG()`, `buildStandalonePCLQInfosForBasePodGang()`, `buildPCSGPackConstraintsAndPCLQsForBasePodGang()`, `doBuildBasePodGangPCLQsAndPCSGPackConstraints()`.

**Replace with:** `buildExpectedPodGangsForPCSReplica(sc *syncContext, pcsReplica int) ([]*podGangInfo, error)`:
- Fetch `PodGangMap` for this replica (named `<pcs-name>-<pcsReplica>`)
- For each tuple in `PodGangMap.Spec.Tuples`: build a `podGangInfo` with the tuple's name and composition
- Return the resulting `[]*podGangInfo`

`getExcessPodGangNames()` requires explicit consideration for the Coherent update case. Since `PodGangMap` under Coherent represents a progressive desired state (not a complete one), a PodGang that is no longer in the current `PodGangMap` tuples does not immediately mean it is excess — it may still hold pods being drained by the orchestrator. The rule is:

- A PodGang is excess if and only if it is **not present in the current `PodGangMap` tuples AND has no pod references** (i.e. all its pods have been moved to new MPGs or deleted). This prevents premature deletion of old BPGs/SPGs mid-drain.
- Under RollingRecreate and steady-state reconciles, `PodGangMap` always represents the complete desired state, so the standard excess check (not in expected set) applies directly.

`getExcessPodGangNames()` must be updated to enforce this condition for the Coherent case: check pod references before marking a PodGang as excess.

**Files:** `operator/internal/controller/podcliqueset/components/podgang/syncflow.go`

---

### Step 5 — PCS reconcilespec: init Coherent update (`operator/internal/controller/podcliqueset/reconcilespec.go`)

In `initUpdateProgress()` (line 138), add a Coherent branch:
```go
if pcs.Spec.UpdateStrategy != nil && pcs.Spec.UpdateStrategy.Type == grovecorev1alpha1.CoherentStrategy {
    pcs.Status.CoherentUpdateProgress = &grovecorev1alpha1.CoherentUpdateProgress{
        UpdateStartedAt: metav1.Now(),
    }
    pcs.Status.UpdatedReplicas = 0
    pcs.Status.CurrentGenerationHash = &newGenerationHash
    return r.setGenerationHashAndUpdateStatus(ctx, pcs, pcsObjectName, newGenerationHash)
    // Do NOT set UpdateProgress — that is RollingRecreate-specific.
}
```

**Files:** `operator/internal/controller/podcliqueset/reconcilespec.go`

---

### Step 6 — PodCliqueSetReplica: strategy dispatch + Coherent orchestration

**Files:** `operator/internal/controller/podcliqueset/components/podcliquesetreplica/podcliquesetreplica.go` (modified) + `coherentupdate.go` (new)

In `Sync()` (line 63), replace the single `isAutoUpdateInProgress` branch with strategy-specific dispatch:

```go
if isCoherentUpdateInProgress(pcs) {
    if err := r.orchestrateCoherentUpdate(ctx, logger, pcs, delWork.pcsIndicesToTerminate); err != nil {
        return err
    }
}
if isRollingRecreateUpdateInProgress(pcs) {
    minAvailableBreachedIndices := slices.Collect(maps.Keys(delWork.minAvailableBreachedConstituents))
    if err := r.orchestrateRollingUpdate(ctx, logger, pcs, delWork.pcsIndicesToTerminate, minAvailableBreachedIndices); err != nil {
        return err
    }
}
```

New helper functions (replace old `isAutoUpdateInProgress`):
```go
func isCoherentUpdateInProgress(pcs *grovecorev1alpha1.PodCliqueSet) bool {
    return pcs.Spec.UpdateStrategy != nil &&
        pcs.Spec.UpdateStrategy.Type == grovecorev1alpha1.CoherentStrategy &&
        pcs.Status.CoherentUpdateProgress != nil &&
        pcs.Status.CoherentUpdateProgress.UpdateEndedAt == nil
}

func isRollingRecreateUpdateInProgress(pcs *grovecorev1alpha1.PodCliqueSet) bool {
    return (pcs.Spec.UpdateStrategy == nil || pcs.Spec.UpdateStrategy.Type == grovecorev1alpha1.RollingRecreateStrategy) &&
        pcs.Status.UpdateProgress != nil &&
        pcs.Status.UpdateProgress.UpdateEndedAt == nil
}
```

#### `orchestrateCoherentUpdate` algorithm (`coherentupdate.go`)

The orchestrator's sole responsibility is state machine progression — it computes the takedown set and updates `CoherentUpdateProgress` in status. Pod creation and PodGang creation are driven by PCLQ/PCSG reconcilers and the PodGang component reading `PodGangMap`. The `PodGangMap` component recomputes tuples on every reconcile reflecting the orchestrator's current progress.

Per reconcile iteration:
```
orchestrateCoherentUpdate(pcs, pcsIndicesToTerminate):
  1. computeCoherentPendingWork(pcs, pcsIndicesToTerminate)
     // pcsIndicesToTerminate carries replica indices being deleted due to scale-in.
     // These replicas must be skipped — attempting to update a replica that is
     // simultaneously being terminated is incorrect and will cause spurious errors.
     // This mirrors the same exclusion in getPCSReplicaInfos for RollingRecreate.
     → for each PCS replica not in pcsIndicesToTerminate:
         fetch PodGangMap for this replica
         check if all pods of standalone PCLQs are at new generation hash AND
               all PCSG-owned PCLQs are at new generation hash
         → produces:
             replicasDone[]    // replicas where all PCLQs and PCSG-owned PCLQs already reflect
                               // the new generation hash — no further update action needed
             replicasPending[] // replicas that still have pods at the old generation hash
                               // and need to go through the takedown+MPG creation cycle

  2. if currentlyUpdating replica is set in CoherentUpdateProgress:
       check if InFlightPodGangs are all available
         (each PodGroup in each PodGang has >= MinReplicas ready pods)
       if not available: requeue
         // Scale-in during this wait is safe. PodGangMap recomputes on the next reconcile
         // and reduces the tuple's pod count; the PCLQ reconciler deletes the excess pod.
         // The availability check uses MinReplicas from the PodGang spec (set at creation
         // time from pclq.Spec.MinAvailable, which is immutable), so the threshold is still
         // correct even if the replica count dropped. The takedown set is always recomputed
         // from live pod state on each reconcile, so no stale state accumulates.
       if available:
         check if all old pods for this replica are gone (iteration complete for this replica)
         if complete: set UpdateEndedAt on replica progress entry, clear currentlyUpdating
         else: compute next takedown set from PodGangMap tuples for the next iteration
               // Next iteration may produce one normal MPG + zero or more Tail-MPGs.
               // The take-down set is the union of old pods displaced across ALL those tuples.
               // InFlightPodGangs captures all PodGang names for this iteration so the
               // orchestrator waits for every one of them (normal MPG + tail-MPGs) to become
               // available before advancing to the next iteration.
               delete old pods in the takedown set
               update InFlightPodGangs in CoherentUpdateProgress with all PodGang names for this iteration
               // CreatedPodGangCount increments once per PodGang created — not once per iteration.
               // A normal iteration creates 1 MPG (count +1). The final tail iteration creates
               // N tail-MPGs simultaneously (count +N). GeneratePodGangName is called once per
               // PodGang using the pre-increment value, then count is incremented before the next call.
               increment PodGangState[replicaIndex].CreatedPodGangCount by number of PodGangs created this iteration, patch status, requeue

  3. if no currentlyUpdating: pick next replica from replicasPending (lowest index first)
       set CoherentUpdateProgress.CurrentlyUpdating entry, patch status, requeue

  4. if replicasPending empty AND no currentlyUpdating:
       set CoherentUpdateProgress.UpdateEndedAt → update complete
```

#### Take-down set computation
The take-down set is derived from the PodGangMap tuples computed for the current iteration,
not independently re-derived from `MinAvailable`. The PodGangMap component decides the MPG
composition for this iteration (one normal MPG + zero or more Tail-MPGs), and the orchestrator
reads those tuples to know exactly how many old pods per PCLQ/PCSG to displace.

- The PodGangMap component computes all PodGang tuples for the current iteration:
  - **Normal iteration** (enough old pods remain to fill a complete MVU): one normal MPG tuple only — exactly `MinAvailable` pods per updated standalone PCLQ + `MinAvailable` PCSG replicas. One MPG is created per iteration until a complete MVU can no longer be formed.
  - **Final tail iteration** (remaining old pods are fewer than a full MVU): all remaining old pods form one or more Tail-MPG tuples, all created simultaneously in a single iteration. There is no gate between tail-MPGs and the preceding normal MPG — they are scheduled independently as their own gangs.
- Take-down set = union of old pods/PCSG-replicas being displaced across **all** tuples in this iteration
- Old pods are selected pending-first, then scheduled, to minimise scheduler disruption
- The orchestrator reads pod counts directly from the PodGangMap tuples — it does not re-read `MinAvailable` independently

#### Identifying takedown candidates
A pod is a takedown candidate based on the PodGang it is associated with (via its `grove.io/podgang` label):

1. Look up the pod's `grove.io/podgang` label → PodGang name → find the matching tuple in PodGangMap by `tuple.Name`
2. If `tuple.PodCliqueSetGenerationHash == newHash` AND `tuple.Name` is NOT in `InFlightPodGangs`
   → pod is associated with an already-completed MPG (e.g. MPG-0 when MPG-1 is being created) → **exclude**
3. If `tuple.Name` IS in `InFlightPodGangs`
   → pod is associated with the current in-flight batch → **exclude**
4. If `tuple.PodCliqueSetGenerationHash == oldHash`
   → old pod not yet displaced → **takedown candidate**

This means `InFlightPodGangs` serves a dual purpose: it is observability for the user AND the mechanism
the orchestrator uses to distinguish "current batch" from "already completed MPGs". The PodGangMap component
must write the next iteration's tuples before the orchestrator computes the takedown set.

#### Tail-MPG naming and scheduling
All tail-MPGs in the final iteration are named via `GeneratePodGangName` — `CreatedPodGangCount` is
incremented once per PodGang created (not once per iteration). For a tail iteration creating N tail-MPGs:
- tail-MPG-0 name uses pre-increment count value, count incremented to count+1
- tail-MPG-1 name uses count+1, count incremented to count+2
- ... and so on

Tail-MPG pods carry no `grove.io/preceding-podgang` annotation and require no special gate-removal
logic. They are scheduled as independent gangs, their scheduling gates are removed by the standard
gate-removal path (same as any other PodGang). The orchestrator simply waits for all tail-MPGs in
`InFlightPodGangs` to become available before marking the replica update complete.

---

### Step 7 — PCLQ controller: use PodGangMap for pod creation (`operator/internal/controller/podclique/`)

#### 7a — `processUpdate()` early-exit for Coherent (`reconcilespec.go` line 72)

```go
if pcs.Spec.UpdateStrategy != nil && pcs.Spec.UpdateStrategy.Type == grovecorev1alpha1.CoherentStrategy {
    // PodGangMap drives pod replacement for Coherent; PCLQ controller skips its own update orchestration.
    return ctrlcommon.ContinueReconcile()
}
```

#### 7b — Pod sync: read PodGangMap to determine pod count and PodGang assignment (`components/pod/syncflow.go`)

In `prepareSyncFlow`, after fetching the PCLQ, fetch the `PodGangMap` for this PCS replica. The pod creation
logic differs based on whether the PCLQ is standalone or owned by a PCSG:

**Standalone PCLQ** — look up by PCLQ name in `tuple.PodCliques`:
1. Find the tuple where `tuple.PodCliques` contains this PCLQ's name
2. Count existing pods with `grove.io/podgang == tuple.Name` AND `PodCliqueSetGenerationHash == tuple.PodCliqueSetGenerationHash` (idempotency guard — handles requeues where some pods were already created)
3. Create delta pods up to `tuple.PodCliques[pclqName]` with `grove.io/podgang: <tuple.Name>` set at creation time and scheduling gate set
4. Tuple quota is a hard ceiling — do not use `spec.replicas` as the creation driver (see eviction note below)

**PCSG-owned PCLQ** — look up by owning PCSG name in `tuple.PodCliqueScalingGroups`:
1. Find the tuple where `tuple.PodCliqueScalingGroups` contains the owning PCSG's name
2. Count existing pods with `grove.io/podgang == tuple.Name` AND `PodCliqueSetGenerationHash == tuple.PodCliqueSetGenerationHash` (same idempotency guard)
3. Create delta pods up to `spec.replicas` (always all pods — the PCSG decides replica granularity, not the PCLQ) with `grove.io/podgang: <tuple.Name>` set at creation time and scheduling gate set
4. No tuple quota ceiling here — `spec.replicas` is always the target for PCSG-owned PCLQs

This replaces the current logic of inheriting `grove.io/podgang` from the PCLQ resource label — pods now get the label directly from the `PodGangMap` tuple at creation time. The `PodCliqueScalingGroups` map in the tuple is sufficient for PCSG-owned PCLQ association — there is no need to enumerate PCSG-owned PCLQs explicitly in `PodCliques`.

**Important (standalone PCLQs only):** Under Coherent updates, the tuple quota is a hard ceiling on pod
creation. Consider this scenario: the orchestrator takes down 2 old pods of PCLQ `F` to place them into
MPG-1 (tuple count = 2). Before the PCLQ reconciler reacts, a third old pod of `F` is evicted due to node
failure. The PCLQ reconciler must still only create 2 new pods (matching the MPG-1 tuple quota), not 3.
The evicted old pod is intentionally not replaced — there is no PodGangMap tuple that accommodates a
new-spec pod for it, and creating one would leave it without a PodGang association. It will be accounted
for in a subsequent iteration when the orchestrator advances the takedown set.

`reconcilestatus.go` is untouched — it generically tracks `UpdatedReplicas` and `CurrentPodTemplateHash` as pods come up.
Under Coherent, `processUpdate()` is skipped but `reconcilestatus.go` still runs on every reconcile and updates
`currentPodCliqueSetGenerationHash`, `updatedReplicas`, `scheduledReplicas` etc. from live pod state.
The coherent orchestrator relies on these fields in `computeCoherentPendingWork` to determine whether all pods
of a PCLQ have converged to the new generation hash — the same fields used by `isPCLQUpdateComplete` for RollingRecreate.

---

### Step 8 — PCSG controller: use PodGangMap for pod creation (`operator/internal/controller/podcliquescalinggroup/`)

#### 8a — `processUpdate()` early-exit for Coherent (`reconcilespec.go` line 71)

Same 3-line guard as Step 7a.

#### 8b — Pod sync: read PodGangMap

Same pattern as Step 7b — PCSG reconciler reads `PodGangMap` tuples to determine how many PCSG replicas belong to each PodGang and creates them with correct `grove.io/podgang` label at creation time.

---

### Step 9 — Guard PCLQ resource label-setting (no podgang label on PCLQ resource itself)

Under Coherent, `grove.io/podgang` must not be set on the PCLQ resource — only on individual pods via `PodGangMap`. The PCLQ resource label would incorrectly propagate a single PodGang name to all pods.

#### 9a — PCS-managed PCLQ (`operator/internal/controller/podcliqueset/components/podclique/podclique.go` line 390)

Do not set `grove.io/podgang` on the PCLQ resource at all when the strategy is Coherent:
```go
if pcs.Spec.UpdateStrategy == nil || pcs.Spec.UpdateStrategy.Type != grovecorev1alpha1.CoherentStrategy {
    labels[apicommon.LabelPodGang] = podGangName
}
```

#### 9b — PCSG-managed PCLQ (`operator/internal/controller/podcliquescalinggroup/components/podclique/podclique.go` line 476)

Same guard — do not set `grove.io/podgang` on the PCLQ resource when the strategy is Coherent.

#### 9c — `getAssociatedPodGangName()` (`operator/internal/controller/podclique/components/pod/syncflow.go` line 109)

When the label is absent, check the strategy before deciding whether to error:
- If strategy is Coherent → return `"", nil` — label is intentionally absent; `grove.io/podgang` is set per-pod via `PodGangMap` at creation time, not inherited from the PCLQ resource
- Otherwise → return error as before — label is always expected to be present for RollingRecreate

---

### Step 10 — CRD regeneration and webhook

Run `make generate manifests`. Check `operator/internal/webhook` for any strategy-type enum validation that needs updating. Register `PodGangMap` CRD in the scheme and controller setup.

---

## File Change Summary

| File | Change |
|---|---|
| `operator/api/core/v1alpha1/podcliqueset.go` | Add `CoherentStrategy`, `CoherentUpdateProgress`, `CoherentReplicaUpdateProgress`, `PodGangReplicaState` types; add `PodGangState` and `CoherentUpdateProgress` to `PodCliqueSetStatus` |
| `operator/api/core/v1alpha1/podgangmap.go` | **New** — `PodGangMap`, `PodGangMapSpec`, `PodGangTuple` CRD types |
| `operator/api/common/namegen.go` | Keep old BPG/SPG name functions; add `GeneratePodGangName` |
| `operator/api/common/namegen_test.go` | Add tests for `GeneratePodGangName` |
| `operator/internal/controller/podcliqueset/components/podgangmap/podgangmap.go` | **New** — `PodGangMap` component; computes and reconciles `PodGangMap` resources |
| `operator/internal/controller/podcliqueset/reconcilespec.go` | Coherent branch in `initUpdateProgress` |
| `operator/internal/controller/podcliqueset/components/podcliquesetreplica/podcliquesetreplica.go` | Strategy-specific dispatch in `Sync()` |
| `operator/internal/controller/podcliqueset/components/podcliquesetreplica/coherentupdate.go` | **New** — `orchestrateCoherentUpdate`, takedown set computation, status updates |
| `operator/internal/controller/podcliqueset/components/podgang/syncflow.go` | Replace BPG+SPG computation with `PodGangMap`-driven `buildExpectedPodGangsForPCSReplica` |
| `operator/internal/controller/podclique/reconcilespec.go` | 3-line Coherent early-exit in `processUpdate` |
| `operator/internal/controller/podclique/components/pod/syncflow.go` | Read `PodGangMap` for pod count/assignment; `getAssociatedPodGangName()` tolerates absent label |
| `operator/internal/controller/podcliquescalinggroup/reconcilespec.go` | 3-line Coherent early-exit in `processUpdate` |
| `operator/internal/controller/podcliquescalinggroup/components/podclique/podclique.go` | Guard `LabelPodGang` — skip for Coherent |
| `operator/internal/controller/podcliqueset/components/podclique/podclique.go` | Guard `LabelPodGang` — skip for Coherent |

## What is NOT changing

- `rollingupdate.go` — untouched
- `gangterminate.go` — untouched
- All `reconcilestatus.go` files (PCLQ, PCSG, PCS) — untouched; fully generic
- `IsAutoUpdateStrategy` utility — no change needed (Coherent is not OnDelete)

---

## Verification

1. Unit tests for `orchestrateCoherentUpdate`: first takedown, MPG creation via PodGangMap, MPG availability wait, tail-MPG, update completion.
2. Unit tests for `PodGangMap` component: all 4 cases (BPG/SPG, MPG, Coherent update, RollingRecreate).
3. Existing `podcliquesetreplica_test.go` must still pass (RollingRecreate regression).
4. Manual smoke test: 2-replica PCS (1 standalone PCLQ + 1 PCSG), trigger image update under Coherent strategy — verify PodGangMap updated each iteration, MPGs created in order, old PodGangs emptied and deleted, pods come up with new image.
5. New PCS deployed under Coherent: verify PodGangMap created before any pods, initial deployment creates MPGs directly.
6. `make test` and `make lint`.
