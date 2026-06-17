import type { Results, Landmark } from "@mediapipe/holistic";

export const FRAMES_PER_SEQUENCE = 30;
export const FEATURES_PER_FRAME = 159; // 53 pontos * 3 coordenadas

/**
 * Normaliza os landmarks em relação ao ponto central dos ombros e
 * retorna um vetor flat de 159 posições (11 tronco, 21 mão esq, 21 mão dir).
 */
export function extractAndNormalizeSpatial(results: Results): number[] {
  const origin = getShoulderCenter(results.poseLandmarks);
  const features: number[] = [];

  // 1. Pose (11 pontos principais: 11,12,13,14,15,16,23,24,19,20,21)
  const poseIndices = [11, 12, 13, 14, 15, 16, 23, 24, 19, 20, 21];
  if (results.poseLandmarks) {
    poseIndices.forEach((idx) => {
      const lm = results.poseLandmarks[idx];
      if (lm) {
        features.push(lm.x - origin.x, lm.y - origin.y, lm.z - origin.z);
      } else {
        features.push(0, 0, 0);
      }
    });
  } else {
    for (let i = 0; i < 11 * 3; i++) features.push(0.0);
  }

  // 2. Mão Esquerda (21 pontos)
  if (results.leftHandLandmarks) {
    results.leftHandLandmarks.forEach((lm) => {
      features.push(lm.x - origin.x, lm.y - origin.y, lm.z - origin.z);
    });
  } else {
    for (let i = 0; i < 21 * 3; i++) features.push(0.0);
  }

  // 3. Mão Direita (21 pontos)
  if (results.rightHandLandmarks) {
    results.rightHandLandmarks.forEach((lm) => {
      features.push(lm.x - origin.x, lm.y - origin.y, lm.z - origin.z);
    });
  } else {
    for (let i = 0; i < 21 * 3; i++) features.push(0.0);
  }

  return features;
}

function getShoulderCenter(poseLandmarks: Landmark[] | undefined) {
  if (!poseLandmarks || !poseLandmarks[11] || !poseLandmarks[12]) {
    return { x: 0, y: 0, z: 0 };
  }
  const lShoulder = poseLandmarks[11];
  const rShoulder = poseLandmarks[12];

  return {
    x: (lShoulder.x + rShoulder.x) / 2,
    y: (lShoulder.y + rShoulder.y) / 2,
    z: (lShoulder.z + rShoulder.z) / 2,
  };
}
