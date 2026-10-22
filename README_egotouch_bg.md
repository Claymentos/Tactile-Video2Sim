# EgoTouch → Genesis avec la reconstruction de Benchmark-Genesis : commandes lancées

Toutes les commandes sont lancées depuis la racine du dépôt (`~/Tactile-Video2Sim`), dans l'env `genesis` sauf mention contraire.
Les scripts `launch/*.sh` choisissent eux-mêmes les envs conda de chaque étape (`launch/config.sh`).
La reconstruction appelle le checkout `~/Benchmark-Genesis/reconstruction` (`BG_RECON_ROOT`) et ses envs conda (`sam3`, `moge`, `recgen`, `genesis`). Il faut donc `conda` dans le PATH.

Les logs de chaque commande sont dans `logs/pipeline/`.

---

## 0. Prérequis : les points 2D WiLoR des épisodes déjà téléchargés

`wilor_hands.json` sert à placer la mocap dans la caméra. Il fait maintenant partie des fichiers téléchargés (`tv2s_data.egotouch.FILES`). Pour les 18 épisodes déjà présents, il a été récupéré ainsi :

```bash
python scripts/data/download_egotouch.py --root ~/EgoTouch \
    --tasks hold_teapot,open_toolbox,pick_up_bottle,pick_up_charger,pick_up_fruit,pick_up_mouse,pick_up_screwdriver,squeeze_pliers,twist_bottle_cap \
    --episodes <les horodatages des épisodes locaux> --per-task 0 --files wilor_hands.json
```

---

## 1. Tournevis (`Workbench/pick_up_screwdriver/20260321_103227_213`)

### 1.1 Entraînement RL sur la référence existante de Benchmark-Genesis
Recette de BG : départs aléatoires ou objet en main.

```bash
TV2S_WANDB=1 SKIP_HANDS=1 SKIP_REFERENCE=1 REFERENCE=data/references/screwdriver_103227/reference.npz \
    N_ENVS=2048 ITERATIONS=2000 TRAIN_ARGS="--env.train_start=random" \
    launch/rl_recon.sh screwdriver_103227_bg ~/EgoTouch/Workbench/pick_up_screwdriver/20260321_103227_213/chest.mp4 \
    > logs/pipeline/screwdriver_103227_bg_train.log 2>&1
```

- Run : `logs/screwdriver_103227_bg/20261021_113159/`.
- Évaluation de tous les checkpoints : `eval_rl_recon/`.
- Meilleur checkpoint en main (`model_best_inhand.pt` = `checkpoint_01975.pt`) :
  - F1 tactile 0,63 (précision 0,82, rappel 0,51) ;
  - objet levé à 35 cm ;
  - tenu 100 % du temps.
- Depuis la frame 0 : F1 0.

### 1.2 Reconstruction de BG relancée depuis ce dépôt, puis référence

```bash
EGOTOUCH=pick_up_screwdriver__20260321_103227_213 SKIP_HANDS=1 SKIP_TRAIN=1 SKIP_EVAL=1 \
    launch/rl_recon.sh et_screwdriver_bg > logs/pipeline/et_screwdriver_bg_recon.log 2>&1
```

- `SKIP_HANDS=1` saute Gemini : l'objet et la frame de référence sont lus dans `data/egotouch/scene/<key>.json`, ici `yellow screwdriver`, frame 0.
- Reconstruction : `data/egotouch/bg_recon/pick_up_screwdriver__20260321_103227_213/`.
- Référence : `data/references/et_screwdriver_bg/reference.npz`.

Ce qui s'est passé pendant cette exécution :

- SAM3, avec le prompt texte `table`, ne trouve la table qu'à partir de la frame 71. La reconstruction s'est d'abord arrêtée sur `empty mask at the reference frame`.
- `reconstruct_bg.sh` choisit alors des points sur la table (`scripts/reconstruction/table_points.py`) et relance SAM3 avec `TABLE_POINTS`. Test isolé du choix des points :
  ```bash
  python scripts/reconstruction/table_points.py --recon data/egotouch/bg_recon/pick_up_screwdriver__20260321_103227_213 \
      --frame 0 --object yellow_screwdriver
  # -> 287,75;549,163;111,222;381,228;434,73
  ```
- Notre `SKIP_HANDS=1` (« pas de Gemini ») était transmis au pipeline de BG, où il veut dire « pas de placement de la mocap ». C'est corrigé. La reprise a été faite à partir du placement des mains :
  ```bash
  EGOTOUCH=pick_up_screwdriver__20260321_103227_213 SKIP_HANDS=1 SKIP_TRAIN=1 SKIP_EVAL=1 \
      RECON_ARGS="SKIP_EXTRACT_AND_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1" \
      launch/rl_recon.sh et_screwdriver_bg > logs/pipeline/et_screwdriver_bg_recon2.log 2>&1
  ```
- Il n'y a pas d'env conda `ecfit` sur cette machine. Les étapes MANO et vidéo de rejeu sont désormais sautées automatiquement ; la référence RL ne les utilise pas. Fin de la reconstruction, toutes les étapes déjà faites étant sautées :
  ```bash
  SKIP_EXTRACT_AND_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_HANDS=1 SKIP_MESH=1 SKIP_OBJECT=1 \
      launch/reconstruct_bg.sh pick_up_screwdriver__20260321_103227_213
  ```
- Construction de la référence : retargeting Wuji, monde avec la table à z = 0, axe de l'objet, pose de saisie, sélection physique.
  ```bash
  EGOTOUCH=pick_up_screwdriver__20260321_103227_213 SKIP_HANDS=1 SKIP_TRAIN=1 SKIP_EVAL=1 \
      launch/rl_recon.sh et_screwdriver_bg > logs/pipeline/et_screwdriver_bg_ref.log 2>&1
  ```

Résultats de la reconstruction :

- **Échelle MoGe** : 0,858, corrigée ; les mains se recalent ensuite à l'échelle 1,000.
- **Mesh** : 16,1 × 2,8 × 2,4 cm, étanche.
- **Main** : relevée de 7,3 cm pour passer au-dessus de la table.
- **Pose de saisie** : F1 du gant 0,53. Aucun des 32 candidats n'est tenu en physique, donc on garde le meilleur de l'optimiseur, comme BG.

C'est la même référence que celle de BG à quelques mm près.

### 1.3 Rejeu cinématique dans Genesis (main Wuji + mesh + articulations mocap)

```bash
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_screwdriver_bg/reference.npz \
    --human --frame 100 --record <scratchpad>/sd_f100.mp4 --no-viewer
```

À la frame 100, pendant la saisie, la trajectoire brute de l'objet (CoTracker + profondeur) passe environ 8 cm à côté du poing.

---

## 2. Marteau (`Workbench/pick_up_hammer/20260321_095941_028`), étape par étape

C'est l'épisode que BG a aussi reconstruit (`~/Benchmark-Genesis/reconstruction/human_videos/touchanything/pick_up_hammer_095941`), ce qui permet de comparer.

### 2.1 Téléchargement depuis Hugging Face (fait)

```bash
python scripts/data/download_egotouch.py --root ~/EgoTouch --tasks pick_up_hammer --episodes 20260321_095941
```

Cela donne 10 fichiers dans `~/EgoTouch/Workbench/pick_up_hammer/20260321_095941_028/`, dont `chest.mp4`, `rokoko_hands.json`, `wilor_hands.json`, `pressure_grids.npz` et `masks.npz`.

### 2.2 Reconstruction de BG : caméra, mains mocap placées dans la caméra, mesh et trajectoire de l'objet (en cours)

```bash
OBJECT=hammer FRAME=0 launch/reconstruct_bg.sh pick_up_hammer__20260321_095941_028 \
    > logs/pipeline/hammer_bg_recon.log 2>&1
```

Pas de fichier de scène Gemini ici : l'objet (`OBJECT=hammer`, le prompt utilisé par BG) et la frame de référence sont donnés à la main.

Étapes, dans l'ordre de `run_touchanything_pipeline.sh` :

1. Frames et masques SAM3 (objet, table).
2. Intrinsèques et profondeur MoGe.
3. Mouvement de caméra : CoTracker sur la table.
4. Mains Rokoko placées dans la caméra, sur les points 2D WiLoR et la profondeur ; la profondeur est remise à l'échelle si besoin.
5. Plan de la table.
6. Mesh RecGen.
7. Trajectoire de l'objet : CoTracker, Kabsch, filtre de Kalman.

À vérifier :

- `hands_overlay_cotracker.mp4` : la mocap projetée sur la vidéo ;
- `video_segmentation/tracked_*.mp4` : les masques ;
- `hammer_points_cotracker.mp4` : le suivi de l'objet ;
- `recgen_out_moge/hammer/posed_mesh.obj` : le mesh.

### 2.3 Retargeting de la main sur la Wuji et référence (à lancer)

La main qui tient l'objet est celle dont le gant s'active.

```bash
EGOTOUCH=pick_up_hammer__20260321_095941_028 OBJECT=hammer SKIP_HANDS=1 SKIP_TRAIN=1 SKIP_EVAL=1 \
    launch/rl_recon.sh et_hammer > logs/pipeline/et_hammer_ref.log 2>&1
```

### 2.4 Visualisation dans Genesis : main Wuji retargetée + mesh de l'objet (à lancer)

```bash
# viewer interactif : main Wuji (orange), mesh du marteau sur sa trajectoire (vert), articulations mocap (rouge)
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human
# la main où la reconstruction l'a mise, avant la surélévation qui la sort de la table
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human --no-lift
# vidéo sans fenêtre
python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human \
    --record outputs/et_hammer_replay.mp4 --no-viewer
```

Autres options : `--speed 0.5`, `--frame N` pour figer une frame.

### 2.5 Entraînement RL (plus tard)

```bash
TV2S_WANDB=1 EGOTOUCH=pick_up_hammer__20260321_095941_028 SKIP_HANDS=1 SKIP_RECON=1 SKIP_REFERENCE=1 \
    TRAIN_ARGS="--env.train_start=random" launch/rl_recon.sh et_hammer
```

---

## Code modifié pour cette chaîne (non commité)

- `launch/reconstruct_bg.sh` (nouveau) : la reconstruction de BG sur un épisode EgoTouch. Il :
  - lit l'objet et la frame dans le fichier de scène ;
  - rattrape la table manquée avec des points ;
  - saute MANO si `ecfit` n'est pas installé.
- `scripts/reconstruction/table_points.py` (nouveau) : points sur la table à la frame de référence. Le masque est ramené d'une frame ultérieure par des homographies de suivi Lucas-Kanade.
- `launch/rl_recon.sh` : nouvelle source `EGOTOUCH=<key>`, qui enchaîne reconstruction, référence, RL et évaluation ; ajout de `SKIP_RECON` et `RECON_ARGS`.
- `scripts/reconstruction/replay_raw_genesis.py` : options `--reference`, `--human`, `--no-lift` et `--frame`.
- `src/tv2s_data/egotouch.py` : ajout de `wilor_hands.json` aux fichiers téléchargés.
- `src/tv2s_data/scene.py` : `manipulated_object`.
- `launch/config.sh` : `BG_RECON_ROOT`.
- `README.md`, `launch/prepare_egotouch.sh` : documentation.
