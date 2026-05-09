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
- `short-generationhash` is a truncated prefix (5 chars) of `pcs.Status.CurrentGenerationHash` — encodes *which update* the MPG belongs to; 5 chars is consistent with Kubernetes ReplicaSet/Pod suffix conventions and sufficient for collision resistance within a single PCS's update history
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
- **Regular reconcile with no update in progress (`CoherentUpdateProgress` is nil):** `computeExpectedPodGangs()` must return existing PodGangs for each replica as-is — no name recomputation, no renaming. This covers two cases:
  1. *Existing PCS with BPGs/SPGs:* preserves them until a coherent update is triggered.
  2. *New PCS already deployed with MPGs:* preserves the existing MPGs unchanged.
  In both cases, scale-out creates new PodGangs for the added replicas (using MPG-based naming with an incremented `CreatedPodGangCount`), and scale-in removes excess PodGangs via the existing excess-detection logic. No renaming ever happens outside of an active update.

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
    // PendingPodGangNames are the names of all PodGangs created in the current iteration
    // (one non-tail MPG + zero or more tail-MPGs) whose availability is being waited on
    // before the next iteration can proceed. Empty when no PodGangs are pending.
    PendingPodGangNames []string `json:"pendingPodGangNames,omitempty"`
}
```

The counter never resets — it increments monotonically across updates and scale-out events, guaranteeing unique PodGang names for a given replica over the lifetime of the PCS. The `short-generationhash` segment in the name provides additional scoping per update event.

---

## Implementation Steps

### Step 1 — API changes (`operator/api/core/v1alpha1/podcliqueset.go`)

1. Add `CoherentStrategy UpdateStrategyType = "Coherent"` constant.
2. Update kubebuilder enum validation: `+kubebuilder:validation:Enum={RollingRecreate,Coherent,OnDelete}`.
3. Add to `PodCliqueSetStatus`:
   - `CoherentUpdateProgress *CoherentUpdateProgress`
   - `PodGangState []PodGangReplicaState` — always-present per-replica PodGang counter (see Counter storage section)
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
    UpdateStartedAt   metav1.Time     `json:"updateStartedAt"`
    UpdateEndedAt    *metav1.Time    `json:"updateEndedAt,omitempty"`
    CurrentlyUpdating []CoherentReplicaUpdateProgress `json:"currentlyUpdating,omitempty"`
}

type CoherentReplicaUpdateProgress struct {
    ReplicaIndex    int32        `json:"replicaIndex"`
    UpdateStartedAt metav1.Time  `json:"updateStartedAt"`
    UpdateEndedAt  *metav1.Time `json:"updateEndedAt,omitempty"`
    // PendingPodGangNames are the names of all PodGangs created in the current iteration
    // (one non-tail MPG + zero or more tail-MPGs) whose availability is being waited on
    // before the next iteration can proceed. Empty when no PodGangs are pending.
    PendingPodGangNames []string `json:"pendingPodGangNames,omitempty"`
}
```
5. Run `make generate` to regenerate deepcopy.

**Files:** `operator/api/core/v1alpha1/podcliqueset.go`

---

### Step 2 — PodGang naming: add unified MPG naming function (`operator/api/common/namegen.go`)

1. **Keep** `GenerateBasePodGangName()`, `CreatePodGangNameFromPCSGFQN()`, `GeneratePodGangNameForPodCliqueOwnedByPodCliqueSet()`, `GeneratePodGangNameForPodCliqueOwnedByPCSG()` — still used by the PodGang sync flow for existing replicas with BPG/SPG topology (no active update).
2. **Add** `GeneratePodGangName(pcsName string, replicaIndex int, shortGenerationHash string, createdPodGangCount int) string` — returns `<pcsName>-<replicaIndex>-<shortGenerationHash>-<createdPodGangCount-1>`.
   - Caller passes the current `PodGangState[replicaIndex].CreatedPodGangCount` value (pre-increment); function subtracts 1 to form the name suffix, reflecting that the count is incremented after creation

**Files:** `operator/api/common/namegen.go`, `operator/api/common/namegen_test.go`

---

### Step 3 — Rewrite PodGang sync flow: unified MPG-based computation (`operator/internal/controller/podcliqueset/components/podgang/syncflow.go`)

This is the largest single change. Replace the two-function BPG+SPG computation with one unified function:

**Remove:** `buildExpectedBasePodGangForPCSReplicas()`, `buildExpectedBasePodGangForPCSReplica()`, `buildExpectedScaledPodGangsForPCSG()`, `doBuildExpectedScaledPodGangForPCSG()`, `buildStandalonePCLQInfosForBasePodGang()`, `buildPCSGPackConstraintsAndPCLQsForBasePodGang()`, `doBuildBasePodGangPCLQsAndPCSGPackConstraints()`.

**Replace with:** `buildExpectedPodGangsForPCSReplica(sc *syncContext, pcsReplica int) ([]*podGangInfo, error)` which computes the expected set of PodGangs for a given PCS replica from two independent inputs:

#### Input 1 — Spec-driven (always applied)

Compute the full set of PodGangs implied by `pcs.Spec` for this replica:
- One PodGang per existing PCS replica: for replicas that already have a PodGang in `existingPodGangs`, return it as-is (preserving BPG/SPG names for old replicas)
- For new replicas added via scale-out (no existing PodGang in `existingPodGangs`): compute a new MPG-named PodGang using `GeneratePodGangName` with `PodGangState[replicaIndex].CreatedPodGangCount`
- For each PCSG in `pcs.Spec`: one PodGang per PCSG replica above `MinAvailable`; for existing PCSG replicas return as-is, for new PCSG replicas added via scale-out compute a new MPG-named PodGang

This input alone is sufficient when no Coherent update is in progress. Scale-in is handled by excess-detection (`getExcessPodGangNames`) which removes any PodGang in `existingPodGangs` not present in the expected set.

#### Input 2 — Update-driven (only when Coherent update is in progress)

Additionally include all PodGangs in `CoherentReplicaUpdateProgress.PendingPodGangNames` for this replica — these are the MPGs created in the current iteration (non-tail + tail-MPGs) that the sync flow must not delete and must ensure exist in the cluster. These are created by `orchestrateCoherentUpdate` (Step 5) and written into `PendingPodGangNames` in status; the sync flow reads them and includes them in `expectedPodGangs` so `createOrUpdatePodGangs` and `getExcessPodGangNames` handle them correctly.

**No counter writes to PCS status from the sync flow.** `PodGangState[replicaIndex].CreatedPodGangCount` is owned and incremented exclusively by `orchestrateCoherentUpdate` (Step 5). The sync flow is read-only with respect to status.

**`getExcessPodGangNames()` is unchanged** — pure name-based comparison. Old BPGs/SPGs appear as excess once they are no longer in the expected set (i.e. their pods have all moved to MPGs) and are deleted naturally.

**Files:** `operator/internal/controller/podcliqueset/components/podgang/syncflow.go`

---

### Step 4 — PCS reconcilespec: init Coherent update (`operator/internal/controller/podcliqueset/reconcilespec.go`)

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

### Step 5 — PodCliqueSetReplica: strategy dispatch + Coherent orchestration

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

**Scale-out/in during update:** The update scope (which PCLQs/PCSGs are being updated) is derived fresh from `pcs.Spec` on every reconcile via `computeMVUScope(pcs)` — no caching in status. A scale-out creates new-spec pods that become "pending association" (scheduling gate + no podgang label) and are picked up for PodGang assignment in the same pass. A scale-in reduces the old pods to process. Both are handled correctly with no stale pre-computed state to invalidate.

Per reconcile iteration:
```
orchestrateCoherentUpdate(pcs, pcsIndicesToTerminate):
  1. computeCoherentPendingWork(pcs, pcsIndicesToTerminate)
     → for each PCS replica not in pcsIndicesToTerminate:
         fetch standalone PCLQs + PCSGs for this replica (live, not cached)
         check if all pods of standalone PCLQs are at new generation hash AND
               all PCSG-owned PCLQs are at new generation hash
         → produces: replicasDone[], replicasPending[]

  2. if currentlyUpdating replica is set in CoherentUpdateProgress:
       check if PendingPodGangNames PodGangs are all available
         (each PodGroup in each PodGang has >= MinReplicas ready pods)
       if not available: requeue
       if available:
         check if all old pods for this replica are gone (iteration complete for this replica)
         if complete: set UpdateEndedAt on replica progress entry, clear currentlyUpdating
         else: compute next takedown set, delete old pods, requeue
               (CreatedPodGangCount is incremented when the next PodGang is created, not here)

  3. if no currentlyUpdating: pick next replica from replicasPending (lowest index first)
       set CoherentUpdateProgress.CurrentlyUpdating entry, patch status, requeue

  4. if replicasPending empty AND no currentlyUpdating:
       set CoherentUpdateProgress.UpdateEndedAt → update complete
```

#### Take-down set computation
- Fetch all pods for each standalone PCLQ in the MVU template (via `componentutils.GetPCLQPods`)
- Separate into `oldPods` (pod template hash != target) and `newPods`
- For each PCSG in MVU template: fetch PCSG replicas; separate into old/new PCSG replicas
- Take-down set = `MinAvailable` old pods per standalone PCLQ (pending-first, then scheduled) + `MinAvailable` old PCSG replicas
- If remaining old pods/replicas < one full MVU: add all remaining → tail case

#### Takedown execution
- Delete each pod in the take-down set
- PCLQ controller recreates them with new spec; new pods automatically get the `grove.io/podgang-pending-creation` scheduling gate via the existing pod sync flow

#### PodGang creation (next reconcile after deletion)
- Detect new schedule-gated pods (no `grove.io/podgang` label) for target PCLQ replicas
- Derive PodGang name deterministically: `GeneratePodGangName(pcs.Name, replicaIndex, shortGenerationHash, createdPodGangCount)`
  - `createdPodGangCount` read from `PodGangState[replicaIndex].CreatedPodGangCount`
  - Name is stable across reconciles for the same iteration — idempotent
- **Patch `grove.io/podgang: <podGangName>` on each pod assigned to this PodGang** (strategic merge patch on `metadata.labels`)
- Build PodGang: PodGroups per the update scope composition rules
- Increment `PodGangState[replicaIndex].CreatedPodGangCount` in status and save all created PodGang names to `CoherentUpdateProgress.CurrentlyUpdating[i].PendingPodGangNames`
- The PCLQ pod syncflow sees the label and removes the scheduling gate on the next reconcile
- For leftover PCSG replicas (tail): patch `grove.io/podgang: <tailPodGangName>` + `grove.io/preceding-podgang: <podGangName>` on tail pods; create tail PodGangs using `CreatedPodGangCount+offset` as the name suffix; add all tail PodGang names to `PendingPodGangNames`; keep their gates until the preceding PodGang is available

---

### Step 6 — PCLQ controller: early-exit for Coherent (`operator/internal/controller/podclique/reconcilespec.go`)

In `processUpdate()` (line 72), add immediately after fetching pcs:

```go
if pcs.Spec.UpdateStrategy != nil && pcs.Spec.UpdateStrategy.Type == grovecorev1alpha1.CoherentStrategy {
    // PCS controller drives all pod replacement for Coherent; PCLQ controller is a no-op.
    return ctrlcommon.ContinueReconcile()
}
```

`reconcilestatus.go` is untouched — it generically tracks `UpdatedReplicas` and `CurrentPodTemplateHash` as pods come up with new spec.

---

### Step 7 — PCSG controller: early-exit for Coherent (`operator/internal/controller/podcliquescalinggroup/reconcilespec.go`)

In `processUpdate()` (line 71), same 3-line guard as Step 6.

---

### Step 8 — 1:N pod-label architecture: guard PCLQ label-setting

**Background:** Today there is a 1:1 PCLQ-to-PodGang mapping. The PCLQ resource gets `grove.io/podgang` set at creation time (via `getLabels()` in the PCLQ component), and all pods of that PCLQ inherit the label. Under Coherent, different pods within the same PCLQ can belong to different PodGangs — so the PCLQ-level label is meaningless and must not be set.

**Rule:** When strategy is Coherent:
- The PCLQ resource does NOT get `grove.io/podgang`
- Pods are created without `grove.io/podgang`
- The orchestrator patches `grove.io/podgang` directly on each pod when assigning it to a PodGang
- The scheduling gate is removed from a pod **only after** the label is patched onto it

#### 8a — Guard PCLQ component `getLabels()` in PCS-managed PCLQ

**File:** `operator/internal/controller/podcliqueset/components/podclique/podclique.go`

At line 310, the PCLQ resource is labeled via `getLabels(pcs, pcsReplica, pclqObjectKey, pclqTemplateSpec, podGangName)`. The `podGangName` argument is generated by `GeneratePodGangNameForPodCliqueOwnedByPodCliqueSet(pcs, pcsReplica)`.

Change: pass an empty string as `podGangName` when `pcs.Spec.UpdateStrategy.Type == CoherentStrategy`. In `getLabels()` (line 390), guard the `LabelPodGang` entry:
```go
if podGangName != "" {
    labels[apicommon.LabelPodGang] = podGangName
}
```

#### 8b — Guard PCLQ component `getLabels()` in PCSG-managed PCLQ

**File:** `operator/internal/controller/podcliquescalinggroup/components/podclique/podclique.go`

Same guard at line 476 — pass empty `podGangName` when strategy is Coherent.

#### 8c — Pod creation: handle absent podgang label

**File:** `operator/internal/controller/podclique/components/pod/pod.go`

At line 306, `getLabels()` always writes `apicommon.LabelPodGang: podGangName`. Since `sc.associatedPodGangName` will be empty for Coherent PCLQs, guard it:
```go
if podGangName != "" {
    labels[apicommon.LabelPodGang] = podGangName
}
```
No change to the call site — `podGangName` will naturally be empty when PCLQ has no `grove.io/podgang` label.

#### 8d — `getAssociatedPodGangName()`: tolerate absent label for Coherent

**File:** `operator/internal/controller/podclique/components/pod/syncflow.go` (line 109)

Currently returns `errCodeMissingPodGangLabelOnPCLQ` if label is absent. For Coherent PCLQs the label is intentionally missing. Change: return `""`, `nil` when the label is absent — the caller (`prepareSyncFlow`, line 79) already stores the result in `sc.associatedPodGangName`; downstream paths that need a name (gate removal, PodGang sync) will check for empty string.

```go
func (r _resource) getAssociatedPodGangName(pclqObjectMeta metav1.ObjectMeta) (string, error) {
    podGangName, ok := pclqObjectMeta.GetLabels()[apicommon.LabelPodGang]
    if !ok {
        return "", nil  // Coherent: label absent until orchestrator assigns pod to a PodGang
    }
    return podGangName, nil
}
```

#### 8e — Gate removal: skip pods without `grove.io/podgang` label

**File:** `operator/internal/controller/podclique/components/pod/syncflow.go` (line 272 in `checkAndRemovePodSchedulingGates`)

Add a guard at the top of the per-pod loop, **before** the existing `slices.Contains(sc.podNamesUpdatedInPCLQPodGangs, p.Name)` check:

```go
if _, hasLabel := p.GetLabels()[apicommon.LabelPodGang]; !hasLabel {
    // Orchestrator has not yet assigned this pod to a PodGang; do not remove gate.
    skippedScheduleGatedPods = append(skippedScheduleGatedPods, p.Name)
    continue
}
```

This ensures the gate is never removed before the orchestrator patches `grove.io/podgang` on the pod.

#### 8f — Orchestrator PodGang assignment: patch pod label then remove gate

This is part of the `orchestrateCoherentUpdate` algorithm in `coherentupdate.go` (Step 5). After computing the PodGang pod composition for a given iteration:

1. For each pod assigned to the PodGang, patch `grove.io/podgang: <podGangName>` directly on the pod (strategic merge patch on `metadata.labels`).
2. Create (or confirm existence of) the PodGang with those pod references.
3. Save all created PodGang names to `CoherentUpdateProgress.CurrentlyUpdating[i].PendingPodGangNames`.
4. The PCLQ pod syncflow will then see the label and proceed to remove the scheduling gate on the next reconcile.

**Init container note:** The DownwardAPI mounts `metadata.labels['grove.io/podgang']` as a file inside the container. When the orchestrator patches the label, the kernel updates the DownwardAPI volume dynamically — the init container's blocking read sees the name without any restart needed.

---

### Step 9 — Tail-MPG gate removal (`operator/internal/controller/podclique/components/pod/syncflow.go`)

For tail-MPG pods (PCSG replicas above `MinAvailable` that don't fill a full MVU), the gate removal must wait until the preceding non-tail MPG is available (all PodGroups have `>= MinReplicas` ready pods), not just labeled.

Add `isPodGangAvailable(ctx, logger, podGangName, namespace) (bool, error)` — modelled on the existing `isBasePodGangScheduled()` pattern (line 319):

```go
func (r _resource) isPodGangAvailable(ctx context.Context, logger logr.Logger, namespace, podGangName string) (bool, error) {
    pg, err := componentutils.GetPodGang(ctx, r.client, podGangName, namespace)
    if err != nil {
        return false, groveerr.WrapError(err, errCodeGetPodGang, component.OperationSync, "...")
    }
    for _, podGroup := range pg.Spec.PodGroups {
        pclq := &grovecorev1alpha1.PodClique{}
        if err = r.client.Get(ctx, client.ObjectKey{Name: podGroup.Name, Namespace: namespace}, pclq); err != nil {
            return false, groveerr.WrapError(err, errCodeGetPodClique, component.OperationSync, "...")
        }
        if pclq.Status.ScheduledReplicas < podGroup.MinReplicas {
            return false, nil
        }
    }
    return true, nil
}
```

Tail PodGang pods carry an annotation `grove.io/preceding-podgang: <precedingPodGangName>` set by the orchestrator when assigning them. In `shouldSkipPodSchedulingGateRemoval()`, check this annotation: if present, call `isPodGangAvailable` for the named preceding PodGang before allowing gate removal.

---

### Step 10 — CRD regeneration and webhook

Run `make generate manifests`. Check `operator/internal/webhook` for any strategy-type enum validation that needs updating.

---

## File Change Summary

| File | Change |
|---|---|
| `operator/api/core/v1alpha1/podcliqueset.go` | Add `CoherentStrategy`, `CoherentUpdateProgress`, `CoherentReplicaUpdateProgress`, `PodGangReplicaState` types; add `PodGangState` and `CoherentUpdateProgress` fields to `PodCliqueSetStatus` |
| `operator/api/common/namegen.go` | Keep old BPG/SPG name functions; add `GeneratePodGangName` |
| `operator/api/common/namegen_test.go` | Update tests for removed/added functions |
| `operator/internal/controller/podcliqueset/reconcilespec.go` | Coherent branch in `initUpdateProgress` |
| `operator/internal/controller/podcliqueset/components/podcliquesetreplica/podcliquesetreplica.go` | Strategy-specific dispatch in `Sync()` |
| `operator/internal/controller/podcliqueset/components/podcliquesetreplica/coherentupdate.go` | **New** — `orchestrateCoherentUpdate`, takedown set, MPG creation, per-pod label patching |
| `operator/internal/controller/podcliqueset/components/podgang/syncflow.go` | Rewrite PodGang computation to unified MPG-based |
| `operator/internal/controller/podclique/reconcilespec.go` | 3-line Coherent early-exit in `processUpdate` |
| `operator/internal/controller/podcliquescalinggroup/reconcilespec.go` | 3-line Coherent early-exit in `processUpdate` |
| `operator/internal/controller/podcliqueset/components/podclique/podclique.go` | Guard `LabelPodGang` in `getLabels()` — skip for Coherent |
| `operator/internal/controller/podcliquescalinggroup/components/podclique/podclique.go` | Guard `LabelPodGang` in `getLabels()` — skip for Coherent |
| `operator/internal/controller/podclique/components/pod/pod.go` | Guard `LabelPodGang` in `getLabels()` — omit if empty |
| `operator/internal/controller/podclique/components/pod/syncflow.go` | `getAssociatedPodGangName()` tolerates absent label; gate removal skips unlabeled pods; `isPodGangAvailable` for tail PodGang gate removal |

## What is NOT changing

- `rollingupdate.go` — untouched
- `gangterminate.go` — untouched
- All `reconcilestatus.go` files (PCLQ, PCSG, PCS) — untouched; fully generic
- `IsAutoUpdateStrategy` utility — no change needed (Coherent is not OnDelete)
- e2e test helpers that reference old naming functions — updated as a consequence of Step 2

---

## Verification

1. Unit tests for `orchestrateCoherentUpdate`: first takedown, MPG creation, MPG availability wait, tail-MPG, update completion.
2. Existing `podcliquesetreplica_test.go` must still pass (RollingRecreate regression).
3. Manual smoke test: 2-replica PCS (1 standalone PCLQ + 1 PCSG), trigger image update under Coherent strategy — verify MPGs created in order, old PodGangs emptied and deleted, pods come up with new image.
4. New PCS deployed under Coherent: verify initial deployment creates MPGs directly with counter-based names, no BPGs/SPGs.
5. `make test` and `make lint`.
