import type { Results, Landmark } from "@mediapipe/holistic";

/**
 * Quantidade de frames (tamanho da janela temporal) necessários para alimentar a rede neural LSTM.
 */
export const FRAMES_PER_SEQUENCE = 30;

/**
 * Quantidade de atributos numéricos (features) passados para a rede em cada frame temporal.
 * É o resultado de 53 pontos espaciais tridimensionais coletados (11 tronco + 21 mão esq + 21 mão dir) multiplicados por 3 coordenadas (x, y, z).
 */
export const FEATURES_PER_FRAME = 159; // 53 pontos * 3 coordenadas

/**
 * Função de Engenharia de Atributos Espaciais.
 * 
 * Recebe o objeto bruto gerado pelo processamento do MediaPipe Holistic e
 * o converte em um vetor matemático unidimensional focado apenas nos pontos que importam
 * para a Língua de Sinais (mãos e parte superior do tronco).
 * Além disso, translada todos os pontos espaciais fazendo o centro dos ombros se tornar a
 * nova origem `(0, 0, 0)`. Isso gera "Invariância a Translação", tornando a IA
 * independente de onde o usuário está enquadrado na câmera.
 * 
 * @param results Objeto Results originário do mediapipe/holistic.
 * @returns Um array numérico plano ("flat") contendo exatamente 159 posições normalizadas.
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

/**
 * Calcula o baricentro (ponto central) entre o ombro esquerdo e direito.
 * Funciona como a "âncora" para normalizar todas as demais coordenadas, garantindo
 * que movimentos sejam processados baseados na posição do corpo, e não nos limites absolutos do vídeo.
 * 
 * @param poseLandmarks Lista de landmarks da pose geral extraída pelo mediapipe.
 * @returns Coordenada 3D `(x, y, z)` central.
 */
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
