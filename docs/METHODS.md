# Methods: DexTaG and Benchmark-Genesis rl_recon on bimanual HumanTouch

This document says what each training method is, what was carried over from its repository, what had
to change for HumanTouch and for two hands, and how the two repos map a human glove onto a robot hand.
Numbers marked *(replay)* come from `launch/compare_tactile_mappings.sh` (no policy involved); the
trained-policy comparison comes from `launch/compare_methods.sh`.

## 1. The data and what it forces

HumanTouch (task X001, "put any object into the basket", 2217 episodes at 60 Hz) gives, in one world
frame: head and wrist tracker poses, two 25-joint hand skeletons, two 460-cell tactile gloves on a 23x20
layout (palm 15x16, one 6x4 patch per finger, 100 unassigned cells), and sub-task annotations.

| issue found in the data | consequence | what the pipeline does |
|---|---|---|
| The world is **Unity's: left-handed and y up**. Read as right-handed z-up, every skeleton has the chirality of the other hand and gravity is not along -z (the table, measured from the head-camera depth with the recorded camera calibration, has its normal along +y). | Retargeted Wuji fingers hyper-extended when the human curled; the simulation's gravity was not the scene's (the hands looked 29 cm apart in height, one support per hand was needed). Two earlier fixes treated the symptom: mirroring the world's y (whole scene mirrored), then mirroring each skeleton in its wrist frame (right chirality, world still mirrored and tilted). | Swap y and z at extraction (a mirror: handedness fixed, z up): positions (x, y, z) -> (x, z, y), quaternions (x, y, z, w) -> (-x, -z, -y, w). Checked: both hands get their label's chirality, the table is horizontal within 3 deg once the camera is registered, and the mocap joints fall inside the gloves' SAM3 masks (median 0 px). |
| Glove cells carry a **drifting offset** (~145 of 460 cells > 0 with an open hand) that also **steps** when the glove settles after a grasp. | A centred running baseline left ~40 phantom contacts after a step. | Baseline = max(past-window, future-window) low percentile (10 s windows), then a margin; contact = cleaned > 8 units. |
| **No object pose or mesh.** Both methods need an object trajectory (DexTaG records it; rl_recon reconstructs it from video). | No object, no manipulation task. | A **proxy cylinder** per grasping hand, inferred from the glove and the retargeted hand (below). Real meshes/poses can replace it in the clip files. |
| The right hand's lowest fingertip is ~25 cm above the left's across episodes. | One table cannot support both hands' objects. | One support plate per hand, put under its object at the grasp onset. |
| The proxy's pose at the grasp onset is its in-palm pose: tilted 20-70 deg for most objects, and its plate is lowered (median 1.8 cm, up to 3 cm) to clear the hand. | Objects toppled or fell 2-5 cm at the reset, before any action (DexTaG fails at 5.8 cm). | Before the grasp the object rests in the nearest stable pose (lying or standing) on its plate, blended into the grasp over 6 frames. |
| After the release nothing is under the object. | Released objects fell 70-90 cm to the floor; "successful" episodes ended with the object 90 cm away. | A placing plate under the release pose, the object's reference settles on it; success requires every object within 6 cm of its reference at the clip end. |
| In some clips a hand overlaps the resting object long before the grasp (e.g. ep2 clip01). | The object was ejected at the reset. | Episodes start at the first frame where the hands clear the resting objects (or in hand). |
| Episodes are long (25-50 s) and chain several grasps, alternating hands, often both hands in contact at once (5-60 % of frames). | DexTaG/rl_recon train on short single-grasp clips. | Clips = one manipulation cycle (`reach_grasp_*` + `transport_*`, or `grasp_lift_basket`), both hands kept, an object slot per hand. |

**Proxy object** (`src/tv2s_data/clips.py`). At the firm-grasp frame (max contact in the first 0.5 s of
the grasp), the glove regions in contact select points on the *retargeted Wuji* hand (contacting
fingertips, the palmar centre if >= 4 palm cells fire). The cylinder's axis runs across the palm (index
knuckle -> little knuckle), its centre sits between palm and fingertips (power grasp) or among the
fingertips (pinch), its radius is the points' distance to the axis (1.2-4.5 cm), its length their spread
+ 4 cm. It is rigid in the palm during the grasp, at rest before, left at the release pose after (its
target is invalid 0.2 s after the release). The support plate is under it at the grasp onset (lowered by
up to 3 cm under the pre-grasp reference hand). Because the proxy is built on the retargeted Wuji hand,
the reference is kinematically consistent with it; it is not the real object, and on wide power grasps
its radius saturates at the bound.

## 2. Shared embodiment and simulation

Both methods drive **two Wuji hands on 6-DoF virtual bases** (rl_recon's floating hand: prismatic x,y,z,
fixed mean wrist rotation, revolute x,y,z, all PD with gravity compensation), at 30 Hz with 4 substeps.
DexTaG's xArm7 was dropped: HumanTouch gives wrist poses, not arm configurations, and two arms would add
an IK/reachability problem the data says nothing about. Each env replays one clip; clips are spread over
envs in Genesis' balanced blocks, since every clip has its own cylinder sizes (heterogeneous entities).

The simulated glove (`src/tv2s_env/tactile/sensor.py`) replaces DexTaG's forked `TactileField` +
`TactileVisualizer`: taxel points on the Wuji links are read against the cylinder's exact SDF,
`distance = SDF(taxel)` (rl_recon's ContactDepthProbe) and `pressure = kn * max(r - distance, 0)`
(DexTaG's penalty spring, r = 4 mm); per glove cell, the smallest distance and the largest pressure.

Every env computes the same metrics whatever the method, so runs are scored identically: contact
precision / recall / F1 (sim distance < 4 mm vs glove contact), DexTaG's glove-image similarity, palm /
fingertip / object errors, success and completion.

## 3. The two training methods

| | DexTaG (`DexTaGBimanualEnv`, `PPO_DEXTAG`) | rl_recon (`ReconBimanualEnv`, `PPO_RECON`) |
|---|---|---|
| goal | a retargeter that tracks the demo while the glove contacts guide it | a physically consistent reconstruction of the demo |
| action (26 / hand) | 6 base deltas + 20 **absolute** finger targets in [-1,1] over the limits, EMA 0.2, random 0-1 step latency | **residual** on the reference at t+1: base +-3 cm / +-0.25 rad, fingers +-0.4 rad |
| actor input | reference at t+1 and 16 future frames (wrist, fingers, fingertips, object), proprio, both **simulated glove images** | phase, joint state, reference 1/2/4 frames ahead, object in palm frame + target + axis, **per-channel soft contact and distance**, reference glove contacts 1 and 5 frames ahead, keypoint errors, last action |
| critic input | + reference glove images, action/proprio history (8), last reward | same as actor |
| network | DexTaG temporal encoder (motion MLP, Conv2D glove encoder with one channel per hand, Conv1D future/history), SiLU MLP 512-512-256-128 | ELU MLP 512-256-128 |
| tactile reward | glove-**image** similarity (per-map normalised, gamma 0.5, 3x3 blur, fingertip-weighted L1 on the reference's contact pixels) + tactile-gated fingertip-surface proximity | **soft contact F1** (sigmoid((r - d)/1.5 mm)) + reach shaping pulling the glove's contact taxels to the surface |
| other rewards (scale x dt) | wrist pos/rot, fingertip pos/rot (per finger), finger dofs, object pos, object axis, support-contact, object velocity-change and power penalties | hand kernels (palm 2 cm / 0.25 rad, 15 keypoints 3 cm), object position hinge (2 cm across / 10 cm along the head-camera ray), object axis hinge (15 deg), action rate |
| episodes | whole clip from frame 0; success at the end; failure on fingertip / object / support-contact | frame 0, random pre-grasp start (50 %), in-grasp RSI start (50 %); terminate on palm 10 cm / keypoints 6 cm / object 20 cm (depth-aware); truncate at the clip end |
| training aids | object guide force decaying over 5000 iterations; perturbation kicks (from 2k to 10k iterations); object mass/friction and init-pose randomisation | hand-anchored object assist (velocity-level, gravity-compensated) decaying to 0 over the first half of training |
| PPO | lr 1e-4 adaptive (KL 0.01, x1.5 / 1.5x / 0.5x), clip 0.1, gamma 0.98, entropy 1e-4, std 1 | lr 5e-4 adaptive (KL 0.01, 2x / 0.5x), clip 0.2 + value clipping, gamma 0.99, std 0.4, obs normaliser (std floor 0.1), near-zero actor output (starts on the reference) |
| default mapping | layout | anatomical |
| default size | 1024 envs x 15000 iterations | 2048 envs x 1500 iterations |

Changes forced by HumanTouch / bimanual: every term has a hand axis and is averaged over the hands it
applies to (object terms over hands whose object target is valid, tactile terms over hands whose glove
reads contact); DexTaG's `rotation_ignore_axis` becomes "track the cylinder's axis only"; rl_recon's
object axis target is the proxy's axis seen from the head camera (unsigned: a cylinder has no tip); the
CoTracker/MoGe depth discount is kept on the head-camera ray. DexTaG's student distillation is not
reproduced (it needs the wrist camera and the arm).

## 4. Mapping the human glove onto the robot

DexTaG and rl_recon solve the same problem differently: put each glove cell somewhere on the robot so
that a simulated reading can be compared with the human one.

**DexTaG: layout mapping + pressure image.** In DexTaG's Genesis fork, tactile points are laid on each
robot link and paired with glove pixels through a 2D layout of the hand (`examples/tactile/tactile_layout.py`);
several points per pixel, the pixel shows its strongest point. The comparison is between *images*:
DexTaG's reward normalises each map by its peak, applies gamma and a blur, and compares only on the
pixels where the human glove reads contact, so it rewards the spatial pattern rather than the pressure.
Here (`build_layout_taxels`): 4 points per cell, finger patch rows split evenly over the three Wuji
phalanges (2 rows each), the palm grid spread over the Wuji palm between the wrist and the knuckles.

**rl_recon: anatomical transfer + binary contacts.** Benchmark-Genesis anchors each glove cell on the
*human* hand model (Digital-Tactile-Twin's MANO taxels) and carries it to the Wuji in coordinates both
hands share: the fraction along the phalanx and the angle around it from the palmar direction (fingers),
affine palm coordinates on (wrist, index MCP, little MCP), then a ray cast onto the collision hulls
Genesis actually collides. Contact is binary per channel (distance < 4 mm), compared with soft F1.
Here (`build_anatomical_taxels`): a finger cell sits at the fraction u of the *human finger's length*
measured from the dataset (median phalanx lengths), which decides the human phalanx, then the same
Wuji phalanx at the same local fraction; one taxel per cell.

**Results** *(replay on episodes 1-20: 123 clips; regenerate with launch/compare_tactile_mappings.sh)*:

| hand | mapping | frames | F1 @4mm | F1 @8mm | F1 @15mm | P @4mm | R @4mm | image similarity (DexTaG reward) | soft F1 + reach (rl_recon reward) |
|---|---|---|---|---|---|---|---|---|---|
| left | layout | 1888 | 0.285 | 0.284 | 0.274 | 0.176 | 0.750 | 0.369 | 0.587 |
| left | anatomical | 1888 | 0.273 | 0.282 | 0.275 | 0.170 | 0.692 | 0.376 | 0.577 |
| right | layout | 3501 | 0.537 | 0.553 | 0.567 | 0.463 | 0.641 | 0.385 | 0.652 |
| right | anatomical | 3501 | 0.532 | 0.552 | 0.563 | 0.466 | 0.620 | 0.384 | 0.645 |

Region-level contact F1 (does the region touch at all), right hand: palm 1.00 / 1.00, thumb 0.91 / 0.91, index 0.91 / 0.81, middle 0.89 / 0.78, ring 0.88 / 0.88, pinky 0.90 / 0.90 (layout / anatomical).

Each tactile reward with the object displaced (cm), right hand, layout placement:

| reward | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| DexTaG image similarity | 0.385 | 0.354 | 0.328 | 0.294 | 0.242 |
| rl_recon soft F1 + reach | 0.652 | 0.644 | 0.612 | 0.537 | 0.308 |

Finger rows (proximal -> tip) land on Wuji link2/3/4 as `2 2 3 3 4 4` (layout) vs `2 2 2 3 3 4` (anatomical, index/middle/ring); the same cell moves 6-10 mm (median), palm cells 0 mm. Full tables: `outputs/tactile_comparison/comparison.md`.

**Reading.**
- *Palm.* The two placements are identical by construction. Affine palm coordinates are invariant to
  the human palm's proportions when the glove is only known by its layout (rl_recon got its palm
  anchors from a measured MANO glove model; HumanTouch provides none), so rl_recon's palm transfer
  reduces to DexTaG's layout.
- *Fingers.* The human proximal phalanx is ~45 % of the finger, so the anatomical mapping puts 3 of the
  6 rows on Wuji's proximal link and 1 on the distal one, where the layout puts 2/2/2: the same cell
  moves 6-10 mm (median). Real glove anchors (a measured glove model, as Digital-Tactile-Twin's) would
  make this choice data-driven instead of an assumption.
- *Agreement with the real glove* barely depends on the placement (contact F1 within ~0.01; the
  anatomical placement loses some region-level agreement on the right index/middle fingers), because the
  replay is dominated by the proxy object and the retargeting: placement is not the bottleneck on this
  data, the object is.
- *Signal model matters more.* rl_recon's soft F1 + reach falls steeply when the object is displaced
  (it measures *whether* the right cells touch), while DexTaG's normalised image similarity is nearly
  flat (scale-invariant and restricted to the reference's contact pixels, it rewards *any* similar
  pattern). DexTaG compensates with its tactile-gated fingertip proximity term and strong object
  tracking terms; rl_recon relies on the tactile term itself to place the contacts.
- *Precision is low* for both (many simulated contacts the glove does not report): the proxies are
  wide, and the glove misses light contacts. Recall is high (0.7-0.8): where the glove fires, the
  reference hand does touch the proxy.

## 5. Recommendations

- For HumanTouch, measure the glove's cell anchors on a hand model once (as Digital-Tactile-Twin did for
  the TouchAnything glove); with only the 2D layout, the two mappings differ only on finger rows.
- Replace the proxy objects with reconstructed ones when possible: Benchmark-Genesis'
  `run_touchanything_pipeline.sh` (SAM3 masks, RecGen mesh, CoTracker + MoGe trajectory) applies to the
  head camera videos; the clip files take a mesh/pose per hand in place of the cylinder.
- Use rl_recon's binary contact F1 when the goal is *where* contacts happen (reconstruction, contact
  transfer), DexTaG's image similarity plus proximity when the goal is a robust tracking controller;
  both are available to both methods through the reward registries.
