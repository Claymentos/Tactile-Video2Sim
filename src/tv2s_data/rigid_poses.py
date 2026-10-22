"""Rigid trajectories from tracked 3D points, the object and camera tracking of launch/reconstruct.sh
(Benchmark-Genesis' track_object_v2s2r.py and kalman_poses.py, unchanged):

  kabsch, kabsch_ransac       the rigid motion carrying one point cloud onto another (RANSAC against points whose
                              depth patch caught the background, or that slid off the object)
  gate_poses                  drop the solves that cannot follow from the previous one (frame-to-frame continuity of
                              a point on the object, not of the translation, which rotation noise swings)
  clean_poses                 gate, interpolate the gaps, then smooth on a sliding window (rotations on the manifold)
  kalman_smooth_poses         constant-velocity Kalman filter + RTS smoother, per-frame noise from the RANSAC inlier
                              count, gaps bridged by prediction; orientation as a multiplicative EKF (a rotation and a
                              body-frame angular velocity, the error a rotation vector), loose innovation gate (a real
                              acceleration -- a lift starting -- surprises a constant-velocity model too)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def kabsch(source, target):
    """Rigid transform taking `source` onto `target` (both (N, 3))."""
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - rotation @ source_centre


def kabsch_ransac(source, target, iterations, threshold, rng):
    """Kabsch with RANSAC, returning (rotation, translation, inlier_mask)."""
    best_inliers = None
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            rotation, translation = kabsch(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        inliers = residual < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return None, None, None
    rotation, translation = kabsch(source[best_inliers], target[best_inliers])
    return rotation, translation, best_inliers


def smooth_on_manifold(translations, rotations, window, confidence=None):
    """Smooth a pose sequence by weighted averaging over a sliding window.

    Rotations are averaged with scipy's Karcher/chordal mean rather than by filtering
    quaternion components independently and renormalising, which is not a manifold
    operation and distorts orientation when the window spans a large turn.

    `confidence` down-weights frames the rigid solve was unsure about, so a pose
    recovered from a handful of inliers is pulled toward its better-observed
    neighbours instead of dragging them toward it.
    """
    half = window // 2
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / (half / 2.0)) ** 2)

    n_frames = len(translations)
    smoothed_translations = np.empty_like(translations)
    smoothed_rotations = []

    for index in range(n_frames):
        neighbours = np.clip(index + offsets, 0, n_frames - 1)
        weights = kernel.copy()
        if confidence is not None:
            weights = weights * np.clip(confidence[neighbours], 1e-3, None)
        weights /= weights.sum()

        smoothed_translations[index] = weights @ translations[neighbours]
        smoothed_rotations.append(rotations[neighbours].mean(weights=weights))

    return smoothed_translations, Rotation.concatenate(smoothed_rotations)


def gate_poses(rotations, translations, valid, max_step, max_turn, centre=None):
    """Drop frames whose solved pose cannot follow from the one before it.

    A near-cylindrical, textureless object leaves rotation about its axis weakly
    observable, so individual frames can solve to wild orientations while still
    reprojecting near the mask. Continuity is the constraint that catches those.

    Continuity, not distance from a reference pose: the object legitimately travels
    far (here the mug is lifted and set back down), so only an implausible jump
    between neighbouring frames indicates a bad solve.

    With `centre` (a point on the object, reference-frame coordinates) the step is
    measured on where that point goes, R c + t, rather than on t. The translation is
    the motion of a frame ~1 m away at the camera, so a few degrees of rotation noise
    swing it by more than max_step while the object stays put: on a TouchAnything
    clip the translation stepped 18.8 cm per frame (median) against 1.9 cm for the
    centre, and gating on it kept 55 of 223 solves instead of 161.
    """
    valid = valid.copy()
    order = np.flatnonzero(valid)
    if len(order) < 3:
        return valid
    if centre is not None:
        translations = np.einsum("tij,j->ti", rotations, np.asarray(centre, dtype=float)) + translations

    anchor = order[0]
    for index in order[1:]:
        gap = index - anchor
        step = np.linalg.norm(translations[index] - translations[anchor])
        turn = np.degrees(
            (Rotation.from_matrix(rotations[anchor]).inv() * Rotation.from_matrix(rotations[index])).magnitude()
        )
        if step > max_step * gap or turn > max_turn * gap:
            valid[index] = False
        else:
            anchor = index
    return valid


def clean_poses(rotations, translations, valid, max_step, max_turn, smooth_window, confidence=None,
                centre=None):
    """Drop physically impossible frames, then interpolate and smooth what remains."""
    from scipy.spatial.transform import Slerp

    valid = gate_poses(rotations, translations, valid, max_step, max_turn, centre=centre)
    order = np.flatnonzero(valid)
    if len(order) < 3:
        return rotations, translations, valid

    span = np.arange(order[0], order[-1] + 1)
    filled_translations = np.stack(
        [np.interp(span, order, translations[order, axis]) for axis in range(3)], axis=1
    )
    slerp = Slerp(order, Rotation.from_matrix(rotations[order]))
    filled_rotations = slerp(span)

    window = min(smooth_window, len(span) - (1 - len(span) % 2))
    if window >= 5:
        filled_translations, filled_rotations = smooth_on_manifold(
            filled_translations, filled_rotations, window, confidence[span] if confidence is not None else None
        )

    rotations = rotations.copy()
    translations = translations.copy()
    rotations[span] = filled_rotations.as_matrix()
    translations[span] = filled_translations
    cleaned = np.zeros_like(valid)
    cleaned[span] = True
    return rotations, translations, cleaned


def _constant_velocity(dt, accel_sigma):
    """Transition and process-noise matrices for a 6-state constant-velocity model.

    Q is the discrete white-noise-acceleration covariance: the process noise is an
    unknown acceleration of scale `accel_sigma` held over the step, which is what
    lets the filter follow a real change of motion instead of fighting it.
    """
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    variance = accel_sigma ** 2
    process = np.zeros((6, 6))
    process[:3, :3] = np.eye(3) * (dt ** 4 / 4.0) * variance
    process[:3, 3:] = np.eye(3) * (dt ** 3 / 2.0) * variance
    process[3:, :3] = process[:3, 3:]
    process[3:, 3:] = np.eye(3) * (dt ** 2) * variance
    return transition, process


def _gain(covariance, noise, innovation, gate):
    """Kalman gain for a direct measurement of the first three states.

    Returns (gain, accepted). The measurement is rejected when the Mahalanobis
    distance of its innovation exceeds `gate`, and a rejected frame is then carried
    by prediction like a frame with no solve at all.

    A consistent measurement sits near sqrt(3), which would suggest gating at 3 or so,
    but the innovations on real takes are heavily tailed by the object's own motion
    rather than by bad solves: a hand starting a lift accelerates far past anything a
    constant-velocity prediction expects. Gating at 3 on grasping_cup/take_006
    rejected 44 of 286 frames -- almost all of them real acceleration -- and the
    filter lagged the lift badly enough to take the silhouette IoU from 0.65 to 0.58.
    So the gate is set loose, to catch a solve that has gone somewhere the motion
    cannot explain at all, and the frame-to-frame jump gate upstream stays the real
    outlier test.
    """
    innovation_covariance = covariance[:3, :3] + np.eye(3) * noise
    distance = innovation @ np.linalg.solve(innovation_covariance, innovation)
    if distance > gate ** 2:
        return None, False
    gain = np.linalg.solve(innovation_covariance, covariance[:3, :]).T
    return gain, True


def _rts_correction(filtered_covariance, predicted_covariance, transition):
    """Backward smoothing gain C = P_filt F^T P_pred^-1, both covariances symmetric."""
    return np.linalg.solve(predicted_covariance, transition @ filtered_covariance).T


def smooth_translations(translations, measured, noise, accel_sigma, gate, dt=1.0):
    """Kalman filter + RTS smoother over positions. Returns (positions, updated)."""
    n_frames = len(translations)
    transition, process = _constant_velocity(dt, accel_sigma)

    states = np.zeros((n_frames, 6))
    covariances = np.zeros((n_frames, 6, 6))
    predicted_states = np.zeros((n_frames, 6))
    predicted_covariances = np.zeros((n_frames, 6, 6))
    updated = np.zeros(n_frames, dtype=bool)

    first = int(np.flatnonzero(measured)[0])
    state = np.concatenate([translations[first], np.zeros(3)])
    covariance = np.diag(np.concatenate([np.full(3, noise[first]), np.full(3, (10.0 * accel_sigma) ** 2)]))
    predicted_states[first] = state
    predicted_covariances[first] = covariance
    states[first] = state
    covariances[first] = covariance
    updated[first] = True

    for index in range(first + 1, n_frames):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        predicted_states[index] = state
        predicted_covariances[index] = covariance

        if measured[index]:
            innovation = translations[index] - state[:3]
            gain, accepted = _gain(covariance, noise[index], innovation, gate)
            if accepted:
                state = state + gain @ innovation
                covariance = covariance - gain @ covariance[:3, :]
                covariance = 0.5 * (covariance + covariance.T)
                updated[index] = True

        states[index] = state
        covariances[index] = covariance

    smoothed = states.copy()
    smoothed_covariances = covariances.copy()
    for index in range(n_frames - 2, first - 1, -1):
        correction = _rts_correction(covariances[index], predicted_covariances[index + 1], transition)
        smoothed[index] = states[index] + correction @ (smoothed[index + 1] - predicted_states[index + 1])
        smoothed_covariances[index] = covariances[index] + correction @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ correction.T

    positions = translations.copy()
    positions[first:] = smoothed[first:, :3]
    return positions, updated


def smooth_rotations(rotations, measured, noise, accel_sigma, gate, dt=1.0):
    """Multiplicative Kalman filter + RTS smoother over orientations.

    The state is a nominal rotation with a body-frame angular velocity; the 6x6
    covariance is over the error [rotation vector, angular velocity] around it, and
    the error is folded back into the nominal after every update so it stays small
    enough for the linearisation to hold.
    """
    n_frames = len(rotations)
    _, process = _constant_velocity(dt, accel_sigma)

    nominal = [Rotation.identity()] * n_frames
    predicted_nominal = [Rotation.identity()] * n_frames
    rates = np.zeros((n_frames, 3))
    predicted_rates = np.zeros((n_frames, 3))
    covariances = np.zeros((n_frames, 6, 6))
    predicted_covariances = np.zeros((n_frames, 6, 6))
    transitions = np.zeros((n_frames, 6, 6))
    updated = np.zeros(n_frames, dtype=bool)

    first = int(np.flatnonzero(measured)[0])
    orientation = Rotation.from_matrix(rotations[first])
    rate = np.zeros(3)
    covariance = np.diag(np.concatenate([np.full(3, noise[first]), np.full(3, (10.0 * accel_sigma) ** 2)]))
    nominal[first] = predicted_nominal[first] = orientation
    covariances[first] = predicted_covariances[first] = covariance
    transitions[first] = np.eye(6)
    updated[first] = True

    for index in range(first + 1, n_frames):
        step = Rotation.from_rotvec(rate * dt)
        # Rotating the error into the propagated frame is the adjoint of the step;
        # the right Jacobian coupling the rate in is taken as identity, which holds
        # for the small per-frame turns left after the jump gate.
        transition = np.eye(6)
        transition[:3, :3] = step.inv().as_matrix()
        transition[:3, 3:] = np.eye(3) * dt
        transitions[index] = transition

        orientation = orientation * step
        covariance = transition @ covariance @ transition.T + process
        predicted_nominal[index] = orientation
        predicted_rates[index] = rate
        predicted_covariances[index] = covariance

        if measured[index]:
            innovation = (orientation.inv() * Rotation.from_matrix(rotations[index])).as_rotvec()
            gain, accepted = _gain(covariance, noise[index], innovation, gate)
            if accepted:
                correction = gain @ innovation
                orientation = orientation * Rotation.from_rotvec(correction[:3])
                rate = rate + correction[3:]
                covariance = covariance - gain @ covariance[:3, :]
                covariance = 0.5 * (covariance + covariance.T)
                updated[index] = True

        nominal[index] = orientation
        rates[index] = rate
        covariances[index] = covariance

    smoothed = list(nominal)
    smoothed_rates = rates.copy()
    smoothed_covariances = covariances.copy()
    for index in range(n_frames - 2, first - 1, -1):
        correction_gain = _rts_correction(covariances[index], predicted_covariances[index + 1], transitions[index + 1])
        error = np.concatenate([
            (predicted_nominal[index + 1].inv() * smoothed[index + 1]).as_rotvec(),
            smoothed_rates[index + 1] - predicted_rates[index + 1],
        ])
        correction = correction_gain @ error
        smoothed[index] = nominal[index] * Rotation.from_rotvec(correction[:3])
        smoothed_rates[index] = rates[index] + correction[3:]
        smoothed_covariances[index] = covariances[index] + correction_gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ correction_gain.T

    matrices = rotations.copy()
    matrices[first:] = Rotation.concatenate(smoothed[first:]).as_matrix()
    return matrices, updated


def kalman_smooth_poses(
    rotations,
    translations,
    valid,
    position_sigma,
    accel_sigma,
    rotation_sigma,
    angular_accel_sigma,
    gate,
    confidence=None,
):
    """Smooth a gated pose sequence with a constant-velocity Kalman filter.

    `valid` marks the frames that were solved; gaps inside the solved span are filled
    by the filter's own prediction rather than by interpolation. `confidence` (the
    RANSAC inlier count) scales each frame's measurement noise, so a pose resting on
    a handful of inliers is trusted less than a well-observed one.

    Returns (rotations, translations, valid), with `valid` covering the solved span
    exactly as clean_poses reports it -- the filter is never extrapolated past the
    last measurement, where a constant-velocity model drifts without bound.
    """
    valid = np.asarray(valid, dtype=bool)
    order = np.flatnonzero(valid)
    if len(order) < 3:
        return rotations, translations, valid

    n_frames = len(translations)
    scale = np.ones(n_frames)
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        reference = np.median(confidence[valid])
        if reference > 0:
            scale = np.clip(reference / np.maximum(confidence, 1e-6), 0.25, 4.0)

    position_noise = (position_sigma ** 2) * scale
    rotation_noise = (np.radians(rotation_sigma) ** 2) * scale

    positions, position_updates = smooth_translations(
        translations, valid, position_noise, accel_sigma, gate
    )
    matrices, rotation_updates = smooth_rotations(
        rotations, valid, rotation_noise, np.radians(angular_accel_sigma), gate
    )

    span = np.arange(order[0], order[-1] + 1)
    smoothed_rotations = rotations.copy()
    smoothed_translations = translations.copy()
    smoothed_rotations[span] = matrices[span]
    smoothed_translations[span] = positions[span]

    rejected = int(valid[span].sum() - np.minimum(position_updates, rotation_updates)[span].sum())
    shift = np.linalg.norm(smoothed_translations[span] - translations[span], axis=1)
    print(f"[kalman] smoothed {len(span)} frames, {int(valid[span].sum())} with a measurement, "
          f"{rejected} rejected by the innovation gate, median correction {1000 * np.median(shift):.1f} mm")

    cleaned = np.zeros_like(valid)
    cleaned[span] = True
    return smoothed_rotations, smoothed_translations, cleaned
