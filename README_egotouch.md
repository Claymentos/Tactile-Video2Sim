# EgoTouch → Genesis → PPO : récapitulatif des commandes

Toutes les commandes se lancent depuis la racine du dépôt (`~/Tactile-Video2Sim`).
Les scripts `launch/*.sh` choisissent eux-mêmes l'env conda de chaque étape (`launch/config.sh`, tout y est surchargeable par variable d'environnement : `EGOTOUCH_ROOT`, `PY_SIM`, `PY_SAM3`, ...).
Les commandes `python ...` directes se lancent dans l'env `genesis`.

Exemple utilisé partout : le marteau, épisode `Workbench/pick_up_hammer/20260321_095941_028`, clé `pick_up_hammer__20260321_095941_028`, run `et_hammer`.

| Étape | Commande | Sortie |
|---|---|---|
| 1. Téléchargement | `scripts/data/download_egotouch.py` | `~/EgoTouch/<scene>/<task>/<stamp>/` |
| 2. Scène (optionnel) | `launch/prepare_egotouch.sh` ou `rl_recon.sh` | `data/egotouch/scene/<key>.json` |
| 3. Reconstruction | `launch/reconstruct.sh` | `data/egotouch/recon/<key>/` |
| 4. Référence | `launch/rl_recon.sh` (`SKIP_TRAIN=1`) ou `scripts/rl/prepare_reference.py` | `data/references/<name>/reference.npz` |
| 5. Entraînement PPO | `launch/rl_recon.sh` ou `scripts/policy/run_ppo_recon.py` | `logs/<name>/<time>/` |
| 6. Évaluation | `scripts/policy/eval_rl_recon.py` | `logs/<name>/<time>/eval_rl_recon/` |
| 7. Visualisation | `scripts/policy/visualize_episode.py`, `replay_raw_genesis.py` | `outputs/episodes/` |

---

## 1. Téléchargement d'un épisode

Dataset public `zhouzhoujy/EgoTouch` (Hugging Face), pas de token nécessaire.

```bash
# les tâches et leur nombre d'épisodes
python scripts/data/download_egotouch.py --list

# un épisode : --episodes prend un horodatage (ou un préfixe)
python scripts/data/download_egotouch.py --root ~/EgoTouch --tasks pick_up_hammer --episodes 20260321_095941

# N épisodes par tâche (--per-task 0 : tous)
python scripts/data/download_egotouch.py --root ~/EgoTouch --tasks pick_up_bottle,twist_bottle_cap --per-task 5

# un fichier manquant seulement
python scripts/data/download_egotouch.py --root ~/EgoTouch --tasks pick_up_hammer --episodes 20260321_095941 --files wilor_hands.json
```

Fichiers utiles : `chest.mp4`, `rokoko_hands.json` (mocap), `wilor_hands.json` (points 2D), `pressure_grids.npz` (gant), `masks.npz`.

## 2. Scène : objet manipulé, main utilisée (optionnel)

Le fichier de scène donne l'objet (prompt SAM3), la frame de référence et `hands_used`. Sans lui, il faut passer `OBJECT=`, `FRAME=` à la reconstruction et `--side` à la référence.

```bash
# extraction + parsing de scène en lot (Gemini : GEMINI_API_KEY ; ou VLM=qwen en local)
launch/prepare_egotouch.sh pick_up_hammer__20260321_095941_028
VLM=qwen launch/prepare_egotouch.sh pick_up_hammer__20260321_095941_028
DOWNLOAD=1 TASKS=pick_up_hammer PER_TASK=5 launch/prepare_egotouch.sh      # télécharge aussi
```

Réglages : `MODEL=gemini-3.6-flash` (ou un Qwen-VL Hugging Face avec `VLM=qwen`), `GEMINI_FPS=4`, `SKIP_PARSE=1`, `OVERWRITE=1`, `OUT=data/egotouch`.
`launch/rl_recon.sh` lance aussi le parsing si le fichier de scène manque (sauf avec `SKIP_SCENE=1`).

## 3. Reconstruction

Frames → masques SAM3 (objet, table) → intrinsèques MoGe et profondeur (MoGe, ou Depth-Anything-3 avec `DEPTH=da3`) → mouvement de caméra (CoTracker sur la table) → mains mocap placées dans la caméra → plan de la table → mesh RecGen → trajectoire de l'objet (CoTracker + Kabsch + Kalman).

```bash
OBJECT=hammer FRAME=0 launch/reconstruct.sh pick_up_hammer__20260321_095941_028 \
    > logs/pipeline/hammer_recon.log 2>&1
```

À vérifier dans `data/egotouch/recon/<key>/` : `hands_wilor_check.mp4`, `hands_overlay_cotracker.mp4`, `video_segmentation/tracked_*.mp4`, `<objet>_points_cotracker.mp4`, `recgen_out_moge/<objet>/posed_mesh.obj`.

| Variable | Rôle |
|---|---|
| `OBJECT="text prompt"`, `FRAME=n` | objet et frame de référence (par défaut : le fichier de scène) |
| `SCENE=file.json` | autre fichier de scène |
| `DEPTH=moge\|da3` | modèle de profondeur (défaut `moge`). `da3` : DA3METRIC-LARGE (`estimate_depth_da3.py`, env `PY_DA3=~/venvs/da3`, `DA3_CHECKPOINT`), écrit dans le même `depth_moge/` ; le modèle utilisé est noté dans `video.json` et relu avec `SKIP_DEPTH=1` |
| `TABLE_POINTS="x,y;x,y"` | points sur la table quand SAM3 la rate (sinon choisis automatiquement) |
| `STATIC_PROMPT=table` | prompt de la surface statique |
| `HAND_SIDES=left` | placer seulement ces mains |
| `HAND_PLACEMENT=smooth\|per-frame\|global` | placement lisse dans le temps (défaut), par frame, ou une seule similitude (BG) |
| `HAND_SCALE_FROM=hands\|fit` | échelle de la profondeur depuis la taille des mains (défaut) ou ajustée (BG) |
| `HAND_FIT_FRAMES=0-300`, `HAND_FIXED_SCALE=1.0`, `HAND_DRIFT_KNOTS=30`, `HAND_KNOT_FRAMES=10`, `HAND_POSE_SMOOTH=4`, `HAND_DEPTH=moge\|pnp`, `SCALE_TOLERANCE=0.02` | réglages fins du placement |
| `HANDS=mocap\|hawor`, `HAND_PROMPT=...`, `HAND_PER_FRAME_DEPTH=0` | mains depuis la vidéo (HaWoR) au lieu de la mocap |
| `HAND_CONTACT=1`, `CONTACT_ARGS="..."` | étape 9 : profondeur des mains corrigée le long du rayon caméra pour que le gant touche l'objet à la saisie et au lâcher (`hand_object_contact.py`) |
| `RECGEN_EROSION=1`, `SAM3_OFFLOAD=1`, `SMOOTH_WINDOW=9` | divers |
| `SKIP_FRAMES=1 SKIP_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_MOCAP=1 SKIP_MESH=1 SKIP_OBJECT=1` | reprendre à une étape |

Exemple, refaire seulement le placement des mains :
```bash
SKIP_FRAMES=1 SKIP_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_MESH=1 SKIP_OBJECT=1 OBJECT=hammer FRAME=0 \
    launch/reconstruct.sh pick_up_hammer__20260321_095941_028
```

Correction de la profondeur des mains par le contact (étape 9 seule, sur une reconstruction faite) :
```bash
SKIP_FRAMES=1 SKIP_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_MOCAP=1 SKIP_MESH=1 SKIP_OBJECT=1 HAND_CONTACT=1 \
    launch/reconstruct.sh pick_up_screwdriver__20260321_103227_213
```
Elle garde le placement d'origine dans `mocap_joints_cotracker_raw.npz` (relançable) et écrit `hand_contact.json`. La référence est ensuite à refaire avec `--overwrite`.

### Profondeur DA3 + correction par le contact (recommandé)

**Le problème.** Les mains mocap sont placées frame par frame sur la profondeur monoculaire, alors que la pose de repos de l'objet vient des premières frames. Avec MoGe, la profondeur dérive dans le temps : sur `pick_up_screwdriver/103227`, le tournevis posé sur la table passe de 121 cm (frame 20) à 102 cm (frame 46). Au premier contact, la main se retrouve à 11,7 cm de l'objet, le long du rayon caméra. Dans la référence, l'objet « se téléporte » alors dans la main (+17 cm à la saisie). Le résiduel du poignet (±3 cm) ne peut pas combler cet écart, et la policy n'attrape jamais l'objet.

**Les deux corrections, qui se cumulent :**
1. **`DEPTH=da3`** : Depth-Anything-3, stable dans le temps. L'écart au contact tombe à 5–6 cm, sans relèvement au-dessus de la table, et la pose de saisie tactile est bien meilleure (F1 0,21 → 0,52 ; 21/32 candidats tenus en physique contre 0/32).
2. **`HAND_CONTACT=1`** : à la première frame de contact du gant, les mains sont rapprochées ou éloignées de la caméra d'un facteur λ, le long du rayon caméra, pour que les doigts soient à 1 cm du mesh. λ est interpolé entre les contacts et reste constant avant le premier et après le dernier. Avec DA3, λ ≈ 1,07 (7 cm). Avec MoGe, il fallait λ ≈ 1,18, ce qui pouvait faire passer la main sous la table.

La référence se construit avec `--clearance-mode frame` : une surélévation globale de la main annulerait la correction.

Recette complète (tournevis). Les frames et les masques SAM3 peuvent être copiés d'une reconstruction MoGe existante, puisqu'ils ne dépendent pas de la profondeur :
```bash
KEY=pick_up_screwdriver__20260321_103227_213
RECON=data/egotouch/recon/${KEY}_da3
DEPTH=da3 HAND_CONTACT=1 launch/reconstruct.sh $KEY $RECON > logs/pipeline/sd_da3_recon.log 2>&1
#   (avec frames + SAM3 recopiés : préfixer SKIP_FRAMES=1 SKIP_SAM3=1)
python scripts/rl/prepare_reference.py --recon $RECON --scene data/egotouch/scene/$KEY.json \
    --out data/references/et_screwdriver_da3_contact --clearance-mode frame --overwrite
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_screwdriver_da3_contact/reference.npz --human
```
`launch/rl_recon.sh` lit toujours la reconstruction dans `data/egotouch/recon/<key>`. Pour une reconstruction rangée ailleurs (`_da3`), construis la référence directement comme ci-dessus, puis entraîne avec `SKIP_RECON=1 SKIP_REFERENCE=1`.

## 4. Référence RL

Retargeting de la main sur la Wuji, monde avec la table à z = 0, axe de l'objet depuis les masques, pose de saisie qui explique le mieux le gant, sélection physique.

```bash
SKIP_SCENE=1 SKIP_RECON=1 SKIP_TRAIN=1 SKIP_EVAL=1 PREP_ARGS="--side left --object-target measured" \
    launch/rl_recon.sh et_hammer pick_up_hammer__20260321_095941_028 > logs/pipeline/et_hammer_ref.log 2>&1

# ou directement
python scripts/rl/prepare_reference.py --recon data/egotouch/recon/pick_up_hammer__20260321_095941_028 \
    --out data/references/et_hammer --side left --object-target measured
```

| Option | Rôle |
|---|---|
| `--scene file.json` / `--side left\|right` | la main : `hands_used` de la scène, sinon `--side` |
| `--object id` | l'objet (défaut : celui qui a une trajectoire et un mesh) |
| `--episode dir` | dossier de l'épisode (défaut `<recon>/source`) |
| `--hand-table-clearance 0.008` | relève la main au-dessus de la table (m ; < 0 : off) |
| `--clearance-mode global\|frame` | une seule surélévation, ou par frame lissée |
| `--object-trajectory keyframes\|tracked` | objet posé / porté par la paume (défaut), ou trajectoire CoTracker + Kalman |
| `--anchor vision\|tactile` | offset de l'objet dans la paume depuis la vidéo ou depuis la pose de saisie tactile |
| `--object-target dense\|measured` | score objet identique partout, ou strict là où il est mesuré et lâche en main |
| `--held-weight 0.5`, `--held-margin 3.0` | `measured` : poids et marges sur les frames en main non mesurées |
| `--overwrite` | refaire le retargeting |

**Recette « V4 »** (`et_hammer_v4`, `et_wrench_v4`) : reconstruction par défaut, référence `--side left --object-target measured`, entraînement par défaut (2048 envs, 2000 itérations).

**Recette « DA3 + contact »** (`et_screwdriver_da3_contact`) : reconstruction `DEPTH=da3 HAND_CONTACT=1`, référence `--clearance-mode frame`, entraînement par défaut (§ 5). À utiliser dès que la main et l'objet ne se rejoignent pas dans le rejeu de la référence.

Vérifier la référence :
```bash
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human --no-lift
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human \
    --record outputs/et_hammer_replay.mp4 --no-viewer          # --frame N, --speed 0.5, --loops 1
python scripts/policy/visualize_episode.py --reference data/references/et_hammer/reference.npz
```

## 5. Entraînement PPO

```bash
TV2S_WANDB=1 SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 N_ENVS=2048 ITERATIONS=2000 \
    launch/rl_recon.sh et_hammer pick_up_hammer__20260321_095941_028 > logs/pipeline/et_hammer_train.log 2>&1

# ou directement
python scripts/policy/run_ppo_recon.py --exp_name=et_hammer --env.data.reference=data/references/et_hammer/reference.npz \
    --num_envs=2048 --runner.total_iterations=2000
```

Sur la référence DA3 + contact. `NAME` donne à la fois la référence (`data/references/NAME/reference.npz`) et le dossier de logs :
```bash
TV2S_WANDB=1 SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 \
    launch/rl_recon.sh et_screwdriver_da3_contact pick_up_screwdriver__20260321_103227_213 \
    > logs/pipeline/et_screwdriver_da3_contact_train.log 2>&1
```

Tout d'un coup (scène, reconstruction, référence, PPO, évaluation ; les étapes faites sont sautées) :
```bash
TV2S_WANDB=1 RECON_ARGS="OBJECT=hammer FRAME=0" PREP_ARGS="--side left" \
    launch/rl_recon.sh et_hammer pick_up_hammer__20260321_095941_028
```

Sortie : `logs/<name>/<time>/` avec `checkpoints/`, `configs/`, `progress.csv`, `tb/`, `videos/` et `eval_rl_recon/`.

### Recette complète : `et_hammer_da3_contact_arm` (marteau, main Wuji sur bras Franka)

Toutes les commandes du run `logs/et_hammer_da3_contact_arm/20261026_103141` (26/10/2026), du téléchargement à l'évaluation. Toutes les options sont écrites, y compris celles laissées à leur valeur par défaut.

- **Code.** Le commit `2356d74` ou suivant. Le run a été lancé sur `5703f82`, plus le placement du bras noté sur la manipulation et le terme `power`, commités ensuite dans `2356d74`.
- **Environnements Python** (`launch/config.sh`) :
  - `genesis` (`PY_SIM`) : téléchargement, reconstruction, référence, PPO ;
  - `lerobot` (`PY_QWEN`) : Qwen-VL ;
  - `~/venvs/da3` (`PY_DA3`) : Depth-Anything-3.
- **Machine du run.** RTX 3080 Laptop (16 Go) : 76 min pour 2000 itérations (environ 34 k fps), sans compter l'évaluation.

**0. Variables (même terminal pour toute la suite)**
```bash
cd ~/Tactile-Video2Sim
TASK=pick_up_hammer
STAMP=20260321_095958_211
KEY=${TASK}__${STAMP}
NAME=et_hammer_da3_contact          # la référence
RUN=${NAME}_arm                     # le dossier de logs
mkdir -p logs/pipeline
```

**1. Téléchargement de l'épisode**
```bash
python scripts/data/download_egotouch.py --root ~/EgoTouch --tasks $TASK --episodes $STAMP
```
L'épisode arrive dans `~/EgoTouch/Workbench/pick_up_hammer/20260321_095958_211/`.

**2. Extraction et scène avec Qwen-VL en local**
```bash
VLM=qwen MODEL=Qwen/Qwen3-VL-4B-Instruct GEMINI_FPS=4 OUT=data/egotouch \
    launch/prepare_egotouch.sh $KEY
cat data/egotouch/scene/$KEY.json
```
Résultat obtenu : `hands_used: left` et l'objet `hammer`.

**3. Reconstruction : profondeur DA3 et correction par le contact**
```bash
DEPTH=da3 DA3_CHECKPOINT=depth-anything/DA3METRIC-LARGE HAND_CONTACT=1 \
HANDS=mocap HAND_SIDES="left right" HAND_PLACEMENT=smooth HAND_POSE_SMOOTH=4 HAND_DEPTH=moge HAND_KNOT_FRAMES=10 \
    launch/reconstruct.sh $KEY data/egotouch/recon/$KEY > logs/pipeline/${NAME}_recon.log 2>&1
```
- **Options.** L'objet (`OBJECT`) et la frame de référence (`FRAME`) viennent du fichier de scène. DA3 tourne avec `--process-res 504`.
- **Résultat.**
  - `video.json` indique `"depth": "da3"`.
  - Dans `hand_contact.json`, la saisie est à la frame 37 : λ appliqué de 1,04, écart main–objet de 13,0 cm avant correction et 9,1 cm après.
  - La prise est tenue des frames 43 à 325.

**4. Référence RL**
```bash
python scripts/rl/prepare_reference.py --recon data/egotouch/recon/$KEY --scene data/egotouch/scene/$KEY.json \
    --out data/references/$NAME --clearance-mode frame --hand-table-clearance 0.008 \
    --object-trajectory keyframes --anchor vision --object-target dense
```
- La main (`left`) vient de `hands_used` ; l'objet (`hammer`) est celui qui a une trajectoire et un mesh.
- Résultat : 397 frames à 30 Hz (13,2 s), une prise.
- Ici la main ne passe jamais sous la table (`hand_dz = 0`), donc `--clearance-mode frame` ou `global` donnent la même référence.

**5. Vérification de la référence**
```bash
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/$NAME/reference.npz --human
```

**6. Test court du placement du bras (optionnel)**
```bash
REFERENCE=data/references/$NAME/reference.npz SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 SKIP_EVAL=1 \
    N_ENVS=256 ITERATIONS=30 TRAIN_ARGS="--env.arm.enabled=True --runner.video_interval=0" \
    launch/rl_recon.sh ${NAME}_armtest $KEY 2>&1 | tee logs/pipeline/${NAME}_armtest.log
rm -r logs/${NAME}_armtest
```
Placement attendu :
```
left Panda at (0.27, -0.56) yaw 144 deg, hand turned -120 deg (240 candidates): IK error to frame 354 0.13 cm / 0.5 deg mean
```

**7. Entraînement PPO et évaluation**
```bash
TV2S_WANDB=1 REFERENCE=data/references/$NAME/reference.npz SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 \
    N_ENVS=2048 ITERATIONS=2000 TRAIN_ARGS="--env.arm.enabled=True" \
    launch/rl_recon.sh $RUN $KEY > logs/pipeline/${RUN}_train.log 2>&1
```
Cette commande lance successivement :
```bash
python scripts/policy/run_ppo_recon.py --exp_name=$RUN --env.data.reference=data/references/$NAME/reference.npz \
    --num_envs=2048 --runner.total_iterations=2000 --env.arm.enabled=True
python scripts/policy/eval_rl_recon.py --exp_name $RUN --all
```

Configuration effective du run, d'après `configs/*.json`. Seuls `data.reference` et `arm.enabled` diffèrent des valeurs par défaut.

| Bloc | Valeurs |
|---|---|
| PPO (`--algo.*`) | MLP 512-256-128 ELU (acteur et critique), `lr` 5e-4 adaptatif (`target_kl` 0.01, bornes 1e-5 à 5e-3), `gamma` 0.99, `gae_lambda` 0.95, `clip_ratio` 0.2, `value_clip` True, `value_loss_coef` 1.0, `entropy_coef` 0, `max_grad_norm` 1.0, 5 epochs × 4 mini-batches, `rollout_length` 32, `initial_log_std` ln 0.4, `obs_normalization` True |
| Runner (`--runner.*`) | `total_iterations` 2000, `log_interval` 10, `save_interval` 25, `video_interval` 500 |
| Env (`--env.*`) | `seed` 3, `substeps` 4, `train_start` first, `eval_start` first, `active_hands` reference, `action_scale_finger` 0.4, `w_tac` 1.5, `w_hand` 1.0, `w_obj` 0.5, `w_obj_axis` 0.75, `w_rate` 0.02, `term_wrist` 0.10, `term_kp` 0.06, `term_obj` 0.20, `assist_init` 1.0, `assist_decay` 0.5, `obs_future` (1,2,4), `obs_tactile_future` (1,5) |
| Bras (`--env.arm.*`) | `enabled` True, placement auto (`base_distance` 0.35…0.65, `base_lateral` −0.2…0.2, `mount_yaw` auto par pas de 30°), `mount_offset` 0.02, `rest_q` (0, −0.3, 0, −2.2, 0, 2.0, 0.785), `kp` (4500, 4500, 3500, 3500, 2000, 2000, 2000), `kv` (450, 450, 350, 350, 200, 200, 200), `effort` (87 ×4, 12 ×3), `action_scale` (0.05 ×4, 0.1 ×3) |
| Récompenses du bras | `w_posture` 0.3 (`sigma_posture` 0.3), `w_limit` 0.5 (`limit_margin` 0.15), `w_velocity` 0.05, `w_torque` 0.05, `w_table` 1.0 (`table_clearance` 0.08), `w_power` 0.4 (`k_power` 0.01) |

**Résultats obtenus**

| | |
|---|---|
| Entraînement, itération 2000 | longueur moyenne 194 frames, tac F1 0.49, puissance ~67 W |
| Évaluation depuis la frame 0, meilleur checkpoint (1875) | tac F1 0.40, poignet 1.6 cm, objet 12.6 cm (3D), hauteur max de l'objet 56 cm, premier échec à la frame 194 |
| Évaluation en main (frame 131) | tac F1 ≈ 0, objet ~16 cm : la prise depuis la main n'est pas apprise |

Le premier échec, à la frame 194, correspond au geste rapide des frames 189–207. Il demande jusqu'à ~3 fois la vitesse max des articulations du Panda (§ « Bras Franka Panda », limite connue).

**8. Vidéo de la policy**
```bash
python scripts/policy/visualize_episode.py --exp $RUN --checkpoint best
```

### Variables de `launch/rl_recon.sh NAME KEY`

| Variable | Rôle |
|---|---|
| `N_ENVS=2048`, `ITERATIONS=2000` | envs parallèles, itérations PPO |
| `TRAIN_ARGS="..."` | options de `run_ppo_recon.py` (tableaux ci-dessous) |
| `REFERENCE=path/reference.npz` | entraîner sur une autre référence (défaut `data/references/NAME/reference.npz`) |
| `RESUME=ckpt.pt`, `FINETUNE=1` | reprise ; avec `FINETUNE=1`, itérations et assistance repartent de 0 |
| `RECON_ARGS="..."`, `PREP_ARGS="..."` | réglages de la reconstruction / de la référence |
| `SKIP_SCENE / SKIP_RECON / SKIP_REFERENCE / SKIP_TRAIN / SKIP_EVAL=1` | sauter des étapes |
| `VLM=qwen`, `MODEL=...` | parsing de scène |
| `TV2S_WANDB=1`, `WANDB_PROJECT`, `WANDB_RUN_GROUP`, `WANDB_MODE=offline` | wandb |

### Options de `run_ppo_recon.py`

Syntaxe `--clé=valeur`, notation pointée pour les configs : `--env.<champ>`, `--env.hands.<champ>`, `--env.tactile.<champ>`, `--algo.<champ>`, `--runner.<champ>`. Tuples séparés par des virgules (`--env.hands.kp_range=0.8,1.2`). Les configs effectives sont écrites dans `logs/<name>/<time>/configs/`.

**Script**

| Option | Défaut | Rôle |
|---|---|---|
| `--exp_name` | requis | nom du run (`logs/<exp_name>/`) |
| `--env_name` | `rl_recon` | env du registre |
| `--num_envs` | 2048 | envs parallèles |
| `--resume_checkpoint` | — | `.pt` ou `latest` |
| `--reset_iterations` | False | fine-tune : réseau seul, itérations à 0 |
| `--lightning` | False | même PPO entraîné par un Trainer Lightning |
| `--device` | `cuda` | |

**Runner (`--runner.*`)**

| Option | Défaut | Rôle |
|---|---|---|
| `total_iterations` | 1500 (`rl_recon.sh` : 2000) | itérations PPO |
| `log_interval` | 10 | stdout, CSV, TensorBoard, wandb |
| `save_interval` | 25 | checkpoints |
| `video_interval` | 500 | vidéo du rollout de la policy sur wandb (`rollout/video`) et dans `videos/` ; 0 : off |

**Départs et épisodes (`--env.*`)**

| Option | Défaut | Rôle |
|---|---|---|
| `train_start` | `first` | `first` : depuis la frame 0 ; `random` : recette BG (au repos ou en main) |
| `start_frame_prob` | 0.5 | `random` : part des départs au repos tirés au hasard (sinon frame 0) |
| `rest_starts` | `before_first_grasp` | départs au repos avant la 1re saisie, ou `away_from_holds` |
| `rest_margin_s` | 0.33 | distance minimale à une saisie (s) |
| `rsi_grasp_prob` | 0.5 | `random` : part des départs en main, objet dans la paume |
| `rsi_min_contact_frac` | 0.5 | départ en main là où le gant atteint cette fraction de son pic |
| `rsi_min_clearance` | 0.003 | ... et l'objet tenu est au-dessus de la table (m) |
| `start_clearance` | -1 | distance taxels / objets au repos aux départs (< 0 : celle de la référence) |
| `episode_window_s` | 0 | `random` : durée des épisodes (0 : jusqu'à la fin) |
| `eval_start` | `first` | évaluation et vidéos depuis la frame 0 ou `in_hand` |
| `term_wrist`, `term_kp`, `term_obj` | 0.10, 0.06, 0.20 | seuils de terminaison (m) |

**Récompense (`--env.*`)**

| Option | Défaut | Rôle |
|---|---|---|
| `tactile_reward` | True | False : ablation sans récompense tactile (toujours loggée) |
| `w_tac`, `w_hand`, `w_obj`, `w_obj_axis`, `w_rate` | 1.5, 1.0, 0.5, 0.75, 0.02 | poids tactile, main, objet, axe de l'objet, lissage des actions |
| `tac_reach_weight`, `tac_reach_scale` | 0.5, 0.02 | r_tac = (1 − w) F1 souple + w approche |
| `sigma_wrist_pos`, `sigma_wrist_rot`, `sigma_kp` | 0.02, 0.25, 0.03 | noyaux de la main (m, rad) |
| `obj_margin_across`, `obj_scale_across` | 0.02, 0.05 | erreur de l'objet en travers du rayon caméra |
| `obj_margin_along`, `obj_scale_along` | 0.10, 0.10 | ... le long du rayon |
| `obj_axis_margin`, `obj_axis_scale` | 0.26, 0.5 | axe long de l'objet (rad) |
| `obj_rot_weight`, `obj_rot_margin` | 0.0, 0.6 | rotation complète (off) |
| `obj_depth_discount` | 3.0 | l'erreur le long du rayon compte 1/x (terminaison) |

**Bras Franka Panda (`--env.arm.*`, off par défaut)**

Chaque main Wuji est montée sur un Panda posé sur la table, à la place de la base virtuelle à 6 DoF (`src/tv2s_env/sim/robots/franka_wuji.py`).
- **Placement automatique, à la construction de l'env.** Par IK sur tout le clip, on essaie 20 positions de base, en arrière de la trajectoire du poignet du côté de la tête (là où était l'épaule du démonstrateur), et 4 orientations de la main sur la bride (0, ±90°, 180°). On garde la combinaison qui suit le mieux la trajectoire. Le résultat est affiché au lancement : `left Panda at (x, y) yaw …, IK error …`.
- **Trajectoire articulaire de référence.** La trajectoire IK du bras sert de référence. Les 7 premières valeurs de l'action, par main, sont un résiduel sur cette trajectoire. Sa vitesse est envoyée en feedforward au PD.
- **Observation.** Elle ajoute les articulations du bras et leur écart à la référence à t+1.
- **Inerties.** Celles de l'URDF Genesis (`panda_bullet`) sont fictives : 0,1 kg m² partout. Elles sont remplacées par celles de MuJoCo Menagerie.

| Option | Défaut | Rôle |
|---|---|---|
| `arm.enabled` | False | True : main sur bras Franka (action 27 par main au lieu de 26) |
| `arm.base_xy`, `arm.base_yaw` | auto | imposer la base (m, rad) au lieu de la chercher |
| `arm.base_distance`, `arm.base_lateral` | (0.35…0.65), (−0.2…0.2) | grille de bases candidates autour du centre de la trajectoire |
| `arm.mount_yaw` | auto | rotation de la main sur la bride (rad) ; auto : le meilleur multiple de 30°, noté par IK sur la manipulation (jusqu'à 1 s après le dernier contact du gant) |
| `arm.mount_offset` | 0.02 | distance bride → paume (m) |
| `arm.rest_q` | pose « ready » du Panda | posture vers laquelle l'IK tire dans l'espace nul |
| `arm.kp`, `arm.kv`, `arm.effort` | gains Genesis, couples max du Panda (87 / 12 N m) | PD et limites |
| `arm.action_scale` | 0.05 ×4, 0.1 ×3 | amplitude du résiduel articulaire (rad) |

Termes de récompense ajoutés quand le bras est activé :

| Terme | Option (poids, échelle) | Rôle |
|---|---|---|
| `arm_posture` | `arm.w_posture` 0.3, `arm.sigma_posture` 0.3 | + exp(−‖q − q_ref‖ / σ) : rester près de la posture IK (coude) |
| `arm_limit` | `arm.w_limit` 0.5, `arm.limit_margin` 0.15 | − profondeur dans les derniers 0.15 rad des butées |
| `arm_velocity` | `arm.w_velocity` 0.05 | − (dq / vitesse max du Panda)² |
| `arm_torque` | `arm.w_torque` 0.05 | − (couple / couple max)² |
| `arm_table` | `arm.w_table` 1.0, `arm.table_clearance` 0.08 | − segments 3 à 7 du bras plus bas que 8 cm au-dessus de la table |
| `power` | `arm.w_power` 0.4, `arm.k_power` 0.01 | + exp(−k Σ\|couple × vitesse\|) sur le bras et les doigts (W) : `AllDOFPowerPenalty` de DexTaG, k recalibré pour le Panda (~50 W en suivi, 300–550 W dans un geste rapide) |

Les métriques correspondantes sont loggées : `e_arm_q`, `p_arm_limit`, `p_arm_velocity`, `p_arm_torque`, `p_arm_table`, `arm_min_z`, `power` (W).

Limite connue : les articulations 5 à 7 du Panda sont plafonnées à 2,6 rad/s et 12 N m. Une rotation rapide du poignet humain (jusqu'à 7 rad/s sur le tournevis, frames 160–190) ne peut pas être suivie à la même vitesse.

**Assistance objet (`--env.*`, entraînement seulement)**

| Option | Défaut | Rôle |
|---|---|---|
| `assist_init` | 1.0 | 0 : pas d'assistance |
| `assist_decay` | 0.5 | décroît à 0 sur cette fraction du run |
| `assist_tau`, `assist_gain` | 0.1, 0.5 | contrôleur virtuel ancré à la main |

**Actions et observations (`--env.*`)**

| Option | Défaut | Rôle |
|---|---|---|
| `action_scale_pos`, `action_scale_rot`, `action_scale_finger` | 0.03, 0.25, 0.4 | amplitude du résiduel (m, rad, rad) |
| `glove_obs` | True | False : contacts du gant à venir mis à zéro |
| `obs_future`, `obs_tactile_future` | (1,2,4), (1,5) | pas futurs de la référence / du gant observés |

**Scène (`--env.*`)**

| Option | Défaut | Rôle |
|---|---|---|
| `data.reference` | requis | `reference.npz` |
| `active_hands` | `reference` | mains simulées : celles de la référence, ou `left` / `right` |
| `tactile_objects` | `manipulated` | ce que le gant peut toucher : l'objet de la main, ou `all` |
| `object_mass` | 0 | kg ; 0 : celle de la référence |
| `object_friction` | 1.0 | |
| `decompose_threshold`, `sdf_voxel` | 0.05, 0.001 | collision convexe, SDF |
| `substeps`, `seed` | 4, 3 | |
| `hands.kp_*`, `hands.kv_*` | 3000/80 base pos, 300/20 rot, 20/1 doigts | gains PD |
| `hands.kp_range`, `hands.kd_range` | (1,1), (1,1) | randomisation des gains (ex. `0.8,1.2`) |
| `hands.friction`, `hands.finger_effort_scale`, `hands.base_force_limit`, `hands.base_torque_limit` | 1.0, 2.0, 200, 20 | |
| `tactile.probe_radius`, `tactile.kn`, `tactile.soft_tau`, `tactile.glove_scale` | 0.004, 250, 0.0015, 30 | gant simulé |

**PPO (`--algo.*`)**, valeurs de la recette rl_recon

| Option | Défaut | Rôle |
|---|---|---|
| `lr`, `value_lr` | 5e-4, None (= lr) | |
| `lr_type` | `ADAPTIVE` | ou `FIXED` |
| `lr_adaptive_factor`, `lr_min`, `lr_max` | 1.5, 1e-5, 5e-3 | |
| `target_kl`, `kl_high`, `kl_low` | 0.01, 2.0, 0.5 | lr ajusté selon le KL |
| `gamma`, `gae_lambda` | 0.99, 0.95 | |
| `clip_ratio`, `value_clip` | 0.2, True | |
| `value_loss_coef`, `entropy_coef`, `max_grad_norm` | 1.0, 0.0, 1.0 | |
| `num_epochs`, `num_mini_batches`, `rollout_length` | 5, 4, 32 | |
| `initial_log_std`, `actor_output_init` | log 0.4, 1e-3 | départ proche du résiduel nul |
| `obs_normalization`, `obs_norm_std_floor` | True, 0.1 | |
| `critic_warmup_iters` | 0 | acteur gelé pendant ces itérations |
| `policy_backbone.hidden_dims`, `critic_backbone.hidden_dims` | 512,256,128 | |
| `policy_backbone.activation` | `ELU` | `RELU`, `TANH`, `SWISH` |

### Modes d'entraînement courants

| Mode | `TRAIN_ARGS` / variables |
|---|---|
| Depuis la frame 0 (défaut, V4) | — |
| Recette BG : départs aléatoires et en main | `--env.train_start=random` |
| Sans récompense tactile | `--env.tactile_reward=False` |
| Sans observation du gant | `--env.glove_obs=False` |
| Sans assistance objet | `--env.assist_init=0` |
| Une seule main | `--env.active_hands=left` |
| Évaluation et vidéos en main | `--env.eval_start=in_hand` |
| Randomisation des gains | `--env.hands.kp_range=0.8,1.2 --env.hands.kd_range=0.8,1.2` |
| Vidéos plus fréquentes / off | `--runner.video_interval=250` / `=0` |
| Lightning | `--lightning=True` |
| Main sur bras Franka Panda | `--env.arm.enabled=True` |
| Fine-tune | `RESUME=logs/et_hammer/<time>/checkpoints/checkpoint_01000.pt FINETUNE=1` |

## 6. Évaluation

Rollouts déterministes, sans assistance ni terminaison, depuis la frame 0 et en main ; copie les meilleurs checkpoints dans `model_best.pt` et `model_best_inhand.pt`.

```bash
python scripts/policy/eval_rl_recon.py --exp_name et_hammer --all            # tous les checkpoints
python scripts/policy/eval_rl_recon.py --exp_name et_hammer --checkpoint 1000 --n-envs 32
```

## 7. Visualisation d'une policy

Vidéo trois panneaux (réel | simulation vue de la caméra | vue de la table, référence en fantôme) et trajectoire simulée en `.npz`.

```bash
python scripts/policy/visualize_episode.py --exp et_hammer --checkpoint best              # best_inhand, latest, N
python scripts/policy/visualize_episode.py --exp et_hammer --checkpoint best_inhand --start in_hand
python scripts/policy/visualize_episode.py --exp et_hammer --run logs/et_hammer/<time> \
    --checkpoint logs/et_hammer/<time>/checkpoints/checkpoint_00500.pt --out outputs/test_video
```

## Jean Zay (SLURM)

Par démonstration à une main (`hands_used` = left ou right dans la scène) : un job perception (scène + référence), puis deux jobs PPO, avec et sans récompense tactile, dans un même groupe wandb (hors ligne).

```bash
ITERATIONS=500 N_ENVS=2048 launch/submit_egotouch.sh study1 pick_up_hammer__20260321_095941_028,<autre_clé>
SKIP_PERCEPTION=1 launch/submit_egotouch.sh study1 @keys.txt                 # références déjà construites
nohup launch/wandb_sync_loop.sh > logs/wandb_sync.log 2>&1 &               # sur un nœud de login : envoi wandb
```
Réglages : `QOS=qos_gpu_h100-t3`, `EXTRA="..."` (options de `run_ppo_recon.py` dans `egotouch_rl.slurm`).

## ACE-Data-0 (vue égocentrique, mains HaWoR)

```bash
python scripts/data/extract_ace.py --root ~/ACE-Data-0 --takes t4_000078 --download        # env lerobot
DATASET=ace OBJECT="orange box" FRAME=130 launch/reconstruct.sh t4_000078
python scripts/rl/prepare_reference.py --recon data/ace/recon/t4_000078 --out data/references/ace_t4_000078 --side right
TV2S_WANDB=1 SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 REFERENCE=data/references/ace_t4_000078/reference.npz \
    launch/rl_recon.sh ace_t4_000078 t4_000078
```
`launch/rl_recon.sh` lit ses chemins de scène et de reconstruction sous `data/egotouch/` : pour ACE, la référence se construit avec `prepare_reference.py` directement.
